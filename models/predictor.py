# predictor.py
import os
import joblib
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any


def load_models():
    """Initialize the model and load necessary data"""
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))  # This gets the models directory
        save_path = os.path.join(current_dir, 'saved')
        data_path = os.path.join(current_dir, 'data')
        
        model_info = {
            'models': {
                'homePoints': {
                    'model': joblib.load(os.path.join(save_path, 'home_points_model.joblib')),
                    'metrics': joblib.load(os.path.join(save_path, 'home_points_metrics.joblib'))
                },
                'awayPoints': {
                    'model': joblib.load(os.path.join(save_path, 'away_points_model.joblib')),
                    'metrics': joblib.load(os.path.join(save_path, 'away_points_metrics.joblib'))
                }
            },
            'imputer': joblib.load(os.path.join(save_path, 'imputer.joblib')),
            'encoders': joblib.load(os.path.join(save_path, 'encoders.joblib')),
            'features': joblib.load(os.path.join(save_path, 'features.joblib'))
        }
        
        # Load processed data
        enhanced_df = pd.read_csv(os.path.join(data_path, 'processed_games.csv'))
        enhanced_df['startDate'] = pd.to_datetime(enhanced_df['startDate'])
        
        return model_info, enhanced_df
            
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        raise

# Load models on import
model_info, enhanced_df = load_models()

def calculate_dynamic_weights(home_team, away_team, home_recent, away_recent, enhanced_df):
    """Calculate dynamic weights based on team characteristics and data quality"""
    weights = {
        'base_strength': 0.4,
        'recent_form': 0.15,
        'h2h': 0.1,
        'conference': 0.25,
        'efficiency': 0.3,
        'rankings': 0.25,
    }
   
    # Calculate conference strength gap
    home_conf = home_recent['home_conference'].iloc[0]
    away_conf = away_recent['away_conference'].iloc[0]
    
    conf_games = enhanced_df[
        (enhanced_df['home_conference'].isin([home_conf, away_conf])) |
        (enhanced_df['away_conference'].isin([home_conf, away_conf]))
    ]
    
    # Adjust weights based on conference strength differential
    if len(conf_games) > 0:
        home_conf_wins = conf_games[
            (conf_games['home_conference'] == home_conf) & 
            (conf_games['homePoints'] > conf_games['awayPoints']) |
            (conf_games['away_conference'] == home_conf) & 
            (conf_games['awayPoints'] > conf_games['homePoints'])
        ].shape[0] / conf_games.shape[0]
        
        conf_diff = abs(home_conf_wins - 0.5)
        weights['conference'] *= (1 + conf_diff)
        weights['h2h'] *= (1 - conf_diff)
    
    # Adjust based on ranking differential if available
    if {'fpi', 'spOverall'}.issubset(enhanced_df.columns):
        home_rank = home_recent[['fpi', 'spOverall']].mean().mean()
        away_rank = away_recent[['fpi', 'spOverall']].mean().mean()
        rank_diff = abs(home_rank - away_rank) / max(enhanced_df[['fpi', 'spOverall']].max().max(), 1)
        
        weights['rankings'] *= (1 + rank_diff)
        weights['recent_form'] *= (1 - rank_diff * 0.5)
    
    # Adjust based on data quality
    home_data_quality = len(home_recent) / 12
    away_data_quality = len(away_recent) / 12
    
    data_quality_factor = (home_data_quality + away_data_quality) / 2
    weights['base_strength'] *= data_quality_factor
    weights['recent_form'] *= data_quality_factor
    
    # Normalize weights to sum to 1
    total = sum(weights.values())
    weights = {k: v/total for k, v in weights.items()}
    
    return weights

def predict_game_score(home_team, away_team, enhanced_df, season=None):
   """
   Predict game score using dynamically calculated weights
   """
   # Get team data
   home_recent = enhanced_df[enhanced_df['homeTeam'] == home_team].sort_values('startDate', ascending=False)
   away_recent = enhanced_df[enhanced_df['awayTeam'] == away_team].sort_values('startDate', ascending=False)
   
   if len(home_recent) == 0 or len(away_recent) == 0:
       return None, None, {}

   # Helper functions for handling NaN values
   def safe_mean(series):
       mean_val = series.mean() 
       return float(mean_val) if pd.notnull(mean_val) else 0
       
   def safe_calc(value, default=0):
       return float(value) if pd.notnull(value) else default

   # Calculate dynamic weights
   WEIGHTS = calculate_dynamic_weights(home_team, away_team, home_recent, away_recent, enhanced_df)

   # Track available columns
   available_cols = {
       'epa': {'home_epa', 'away_epa', 'home_epaAllowed', 'away_epaAllowed'}.issubset(enhanced_df.columns),
       'rankings': {'fpi', 'spOverall'}.issubset(enhanced_df.columns),
       'weather': {'temperature', 'windSpeed'}.issubset(enhanced_df.columns),
       'rest': 'rest_days' in enhanced_df.columns
   }

   # Base metrics with NaN handling
   home_avg = safe_mean(enhanced_df[enhanced_df['homeTeam'] == home_team]['homePoints'])
   away_avg = safe_mean(enhanced_df[enhanced_df['awayTeam'] == away_team]['awayPoints'])
   home_advantage = safe_calc(safe_mean(enhanced_df['homePoints']) - safe_mean(enhanced_df['awayPoints']))
   
   # Team performance metrics with NaN handling
   home_strength = safe_calc(safe_mean(home_recent['homePoints']) - safe_mean(enhanced_df['homePoints']))
   away_strength = safe_calc(safe_mean(away_recent['awayPoints']) - safe_mean(enhanced_df['awayPoints']))
   
   # Recent form (weighted last 5 games)
   weights = [0.35, 0.25, 0.2, 0.15, 0.05]
   home_games = home_recent.head(5)['homePoints'].fillna(0).tolist()
   away_games = away_recent.head(5)['awayPoints'].fillna(0).tolist()
   
   home_form = safe_calc(
       np.average(home_games[:len(weights)], weights=weights[:len(home_games)]) - home_avg 
       if home_games else 0
   )
   away_form = safe_calc(
       np.average(away_games[:len(weights)], weights=weights[:len(away_games)]) - away_avg 
       if away_games else 0
   )
   
   # Head-to-head history with reduced importance for cross-conference
   h2h = enhanced_df[
       ((enhanced_df['homeTeam'] == home_team) & (enhanced_df['awayTeam'] == away_team)) |
       ((enhanced_df['homeTeam'] == away_team) & (enhanced_df['awayTeam'] == home_team))
   ].sort_values('startDate', ascending=False)
   
   h2h_factor = 0
   if len(h2h) >= 3:
       h2h_weights = [0.5, 0.3, 0.2]
       h2h_games = h2h.head(3)
       h2h_factor = safe_calc(sum(
           weight * safe_calc(game['homePoints'] - game['awayPoints'])
           for weight, (_, game) in zip(h2h_weights, h2h_games.iterrows())
       ) / len(h2h_games))
       if h2h.iloc[0]['homeTeam'] == away_team:
           h2h_factor *= -1
           
   # Conference stats with enhanced weighting
   home_conf = home_recent['home_conference'].iloc[0]
   away_conf = away_recent['away_conference'].iloc[0]
   
   home_conf_data = enhanced_df[enhanced_df['home_conference'] == home_conf]
   away_conf_data = enhanced_df[enhanced_df['away_conference'] == away_conf]

   conf_stats = {
       'home': {
           'points': safe_mean(home_conf_data['homePoints']),
           'offense': safe_mean(home_conf_data['home_epa']) if available_cols['epa'] else 0,
           'defense': safe_mean(home_conf_data['home_epaAllowed']) if available_cols['epa'] else 0
       },
       'away': {
           'points': safe_mean(away_conf_data['awayPoints']),
           'offense': safe_mean(away_conf_data['away_epa']) if available_cols['epa'] else 0,
           'defense': safe_mean(away_conf_data['away_epaAllowed']) if available_cols['epa'] else 0
       }
   }
   
   conf_diff = safe_calc(
       (conf_stats['home']['points'] - conf_stats['away']['points']) * WEIGHTS['conference'] +
       (conf_stats['home']['offense'] - conf_stats['away']['offense']) * (WEIGHTS['conference'] * 0.5) -
       (conf_stats['home']['defense'] - conf_stats['away']['defense']) * (WEIGHTS['conference'] * 0.5)
   )
   
   # Team efficiency metrics with increased weight
   efficiency_diff = 0
   if available_cols['epa']:
       home_eff = {
           'offense': safe_mean(home_recent['home_epa']),
           'defense': safe_mean(home_recent['home_epaAllowed'])
       }
       away_eff = {
           'offense': safe_mean(away_recent['away_epa']),
           'defense': safe_mean(away_recent['away_epaAllowed'])
       }
       
       efficiency_diff = safe_calc(
           (home_eff['offense'] - away_eff['defense']) * WEIGHTS['efficiency'] -
           (away_eff['offense'] - home_eff['defense']) * WEIGHTS['efficiency']
       )
           
   # Rankings impact with increased weight
   rankings_impact = 0
   if available_cols['rankings']:
       home_rank = safe_mean(home_recent[['fpi', 'spOverall']].mean())
       away_rank = safe_mean(away_recent[['fpi', 'spOverall']].mean())
       rankings_impact = safe_calc((home_rank - away_rank) * WEIGHTS['rankings'])
       
   # Weather effects
   weather_impact = 0
   if available_cols['weather']:
       temp = safe_calc(enhanced_df['temperature'].iloc[0], 70)
       wind = safe_calc(enhanced_df['windSpeed'].iloc[0], 0)
       
       if temp < 40:
           weather_impact -= 1
       elif temp > 85:
           weather_impact -= 0.5
       if wind > 15:
           weather_impact -= wind * 0.1
           
   # Season progress
   season_factor = 0
   if season and 'week' in enhanced_df.columns:
       current_week = safe_calc(enhanced_df[enhanced_df['season'] == season]['week'].max(), 7)
       season_progress = current_week / 14  # Assuming 14 week season
       season_factor = safe_calc(season_progress * 2)
   
   # Rest differential    
   rest_advantage = 0
   if available_cols['rest']:
       home_rest = safe_calc(home_recent['rest_days'].iloc[0] if not home_recent.empty else 7)
       away_rest = safe_calc(away_recent['rest_days'].iloc[0] if not away_recent.empty else 7)
       rest_advantage = safe_calc((home_rest - away_rest) * 0.5)
       
   # Calculate final scores with enhanced weights
   home_score = safe_calc(
       home_avg +
       home_strength * WEIGHTS['base_strength'] +
       home_form * WEIGHTS['recent_form'] +
       h2h_factor * WEIGHTS['h2h'] +
       home_advantage +
       conf_diff +
       efficiency_diff +
       rankings_impact +
       weather_impact +
       season_factor +
       rest_advantage 
   )
   
   away_score = safe_calc(
       away_avg +
       away_strength * WEIGHTS['base_strength'] +
       away_form * WEIGHTS['recent_form'] -
       h2h_factor * WEIGHTS['h2h'] -
       conf_diff -
       efficiency_diff -
       rankings_impact +
       weather_impact +
       season_factor -
       rest_advantage
   )
   
   # Format details without NaN values
   details = {
       'base_scores': {'home': safe_calc(home_avg), 'away': safe_calc(away_avg)},
       'strength': {'home': safe_calc(home_strength), 'away': safe_calc(away_strength)},
       'form': {'home': safe_calc(home_form), 'away': safe_calc(away_form)},
       'h2h_factor': safe_calc(h2h_factor),
       'conf_stats': conf_stats,
       'efficiency': {'diff': safe_calc(efficiency_diff)},
       'rankings': {'impact': safe_calc(rankings_impact)},
       'weather': {'impact': safe_calc(weather_impact)},
       'season': {'factor': safe_calc(season_factor)},
       'rest': {'advantage': safe_calc(rest_advantage)},
       'weights': WEIGHTS
   }
   
   return round(home_score, 1), round(away_score, 1), details