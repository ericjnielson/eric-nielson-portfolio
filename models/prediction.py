# %% [markdown]
# # Import Libraries

# %%
# # College Football Score Prediction Model
# This notebook implements machine learning models to predict home points, away points, and spread in college football games.

### 1. Import Dependencies

# Utilities - Import logging first
import logging
import warnings
from datetime import datetime, timedelta
import joblib

# Data manipulation and analysis
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Union
import asyncio
from gql import gql, Client
from gql.transport.aiohttp import AIOHTTPTransport
from sklearn.ensemble import StackingRegressor
from sklearn.feature_selection import SelectFromModel
from xgboost import XGBRegressor
from sklearn.base import clone
from dotenv import load_dotenv
import os

# Visualization
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme()

# Machine Learning - Core
from sklearn.model_selection import (
    train_test_split,
    cross_val_score, 
    GridSearchCV,
    TimeSeriesSplit
)

# Preprocessing
from sklearn.preprocessing import (
    StandardScaler,
    LabelEncoder,
    RobustScaler,
    OneHotEncoder
)
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Metrics
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    mean_absolute_error,
    make_scorer
)

# Models
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    VotingRegressor
)
from sklearn.linear_model import (
    Ridge,
    ElasticNet,
    Lasso
)

import sys
import subprocess
import pkg_resources

def install_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

try:
    sklearn_version = pkg_resources.get_distribution('scikit-learn').version
    if sklearn_version.startswith('1.6'):
        print("Downgrading scikit-learn to version 1.5.2...")
        install_package('scikit-learn==1.5.2')
        print("Please restart your kernel to use the new scikit-learn version")
except:
    print("Installing scikit-learn version 1.5.2...")
    install_package('scikit-learn==1.5.2')
    print("Please restart your kernel to use the new scikit-learn version")
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    logger.warning("XGBoost not installed. XGBoost models will not be available.")
    XGBOOST_AVAILABLE = False

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Type aliases for clarity
ArrayLike = Union[np.ndarray, pd.Series, List]
DataFrameLike = Union[pd.DataFrame, pd.Series]

# Suppress warnings
warnings.filterwarnings('ignore')

# %% [markdown]
# # API Setup

# %%
dotenv_path = os.path.join(os.path.dirname(os.getcwd()), '.env')
load_dotenv(dotenv_path)

CFB_API_KEY = os.getenv('CFB_API_KEY')

transport = AIOHTTPTransport(
    url="https://graphql.collegefootballdata.com/v1/graphql",
    headers={"Authorization": f"Bearer {CFB_API_KEY}"}
)

client = Client(transport=transport, fetch_schema_from_transport=True)

# %% [markdown]
# # Import Data

# %%
import asyncio

query = gql("""
query gamesQuery($startYear: smallint!, $endYear: smallint!, $offset: Int!) {
  adjustedTeamMetrics {
    teamId
    epa
    epaAllowed
    explosiveness
    explosivenessAllowed
    lineYards
    lineYardsAllowed
    openFieldYards
    openFieldYardsAllowed
    passingDownsSuccess
    passingDownsSuccessAllowed
    passingEpa
    passingEpaAllowed
    rushingEpa
    rushingEpaAllowed
    secondLevelYards
    secondLevelYardsAllowed
    standardDownsSuccess
    standardDownsSuccessAllowed
    success
    successAllowed
    team {
      school
      conference
      classification
    }
    year
  }
  gameLines {
    gameId
    moneylineAway
    moneylineHome
    overUnder
    overUnderOpen
    spread
    spreadOpen
  }
  gameWeather {
    gameId
    temperature
    humidity
    windSpeed
    precipitation
    pressure
    snowfall
    totalSun
    weatherConditionCode
    windDirection
    windGust
  }
  teamTalent {
    talent
    team {
      teamId
      school
      conference
    }
    year
  }
  currentTeams {
    teamId
    school
    conference
    classification
    division
  }
  ratings(    where: {_and: [{year: {_gte: $startYear}}, {year: {_lte: $endYear}}]}
) {
    conference
    conferenceId
    elo
    fpi
    fpiAvgWinProbabilityRank
    fpiDefensiveEfficiency
    fpiGameControlRank
    fpiOffensiveEfficiency
    fpiOverallEfficiency
    fpiRemainingSosRank
    fpiResumeRank
    fpiSosRank
    fpiSpecialTeamsEfficiency
    fpiStrengthOfRecordRank
    spDefense
    spOffense
    spOverall
    spSpecialTeams
    srs
    team
    teamId
    year
  }
  teamTalent {
    talent
    year
    team {
      teamId
      school
      conference
    }
  }
  game(
    where: {_and: [{season: {_gte: $startYear}}, {season: {_lte: $endYear}}]}
    offset: $offset
    limit: 1000
    orderBy: [{season: ASC}]
  ) {
    id
    season
    week
    homeTeam
    homePoints
    homeTeamId
    homeTeamInfo {
      school
      conference
    }
    awayTeam
    awayPoints
    awayTeamId
    awayTeamInfo {
      school
      conference
    }
    startDate
    neutralSite
    seasonType
    attendance
    excitement
    homeStartElo
    homeEndElo
    awayStartElo
    awayEndElo
    homePostgameWinProb
    awayPostgameWinProb
    homeLineScores
    awayLineScores
    conferenceGame
  }
}
""")

async def fetch_all_games():
   all_games = []
   all_data = {}
   offset = 0
   first_batch = True
   validation_info = {}

   while True:
       variables = {
           "startYear": 2008,
           "endYear": 2024, 
           "offset": offset
       }
       
       try:
           result = await client.execute_async(query, variable_values=variables)
           
           if first_batch:
            all_data = {
                'gameLines': result.get('gameLines', []),
                'gameWeather': result.get('gameWeather', []),
                'adjustedTeamMetrics': result.get('adjustedTeamMetrics', []),
                'teamTalent': result.get('teamTalent', []),
                'currentTeams': result.get('currentTeams', []),
                'ratings': result.get('ratings', []), 
                'predictedPoints': result.get('predictedPoints', []) 
            }
            
            first_batch = False

           games_batch = result.get('game', [])
           
           if not games_batch:  
               break
               
           all_games.extend(games_batch)
           print(f"Fetched {len(games_batch)} games (years: {min(g['season'] for g in games_batch)}-{max(g['season'] for g in games_batch)}), total so far: {len(all_games)}")
           
           if len(games_batch) < 1000: 
               break
               
           offset += 1000
           
       except Exception as e:
           print(f"Error fetching games at offset {offset}: {str(e)}")
           if not all_games:  # If we haven't successfully fetched any games
               raise  # Re-raise the exception
           break  # Otherwise break and return what we have

   # Combine all data
   all_data['game'] = all_games

   # Data validation 
   print("\nRaw API Response:")
   validation_info['metrics_years'] = sorted(set(m['year'] for m in all_data['adjustedTeamMetrics']))
   validation_info['games_years'] = sorted(set(g['season'] for g in all_data['game']))
   validation_info['talent_years'] = sorted(set(t['year'] for t in all_data['teamTalent']))
   print("\nMetrics years available:", validation_info['metrics_years'])
   print("\nGames years available:", validation_info['games_years']) 
   print("\nTeam Talent years available:", validation_info['talent_years'])

   print("\nTotal counts in raw data:")
   print(f"Metrics records: {len(all_data['adjustedTeamMetrics'])}")
   print(f"Games records: {len(all_data['game'])}")
   print(f"Team Talent records: {len(all_data['teamTalent'])}")
   print(f"Game Lines records: {len(all_data['gameLines'])}")
   print(f"Game Weather records: {len(all_data['gameWeather'])}")

   # Season distribution
   season_counts = {}
   for game in all_data['game']:
       season = game['season']
       season_counts[season] = season_counts.get(season, 0) + 1
   validation_info['season_counts'] = dict(sorted(season_counts.items()))
   print("\nUnique season values in games:")
   print("Season distribution:", validation_info['season_counts'])

   # Data completeness checks
   games_with_both_scores = sum(1 for g in all_data['game'] if g['homePoints'] is not None and g['awayPoints'] is not None)
   print("\nData completeness checks:")
   print(f"Games with both scores: {games_with_both_scores} ({games_with_both_scores/len(all_data['game'])*100:.1f}%)")

   games_with_weather = sum(1 for g in all_data['gameWeather'])
   print(f"Games with weather data: {games_with_weather} ({games_with_weather/len(all_data['game'])*100:.1f}%)")

   games_with_lines = sum(1 for g in all_data['gameLines'])
   print(f"Games with betting lines: {games_with_lines} ({games_with_lines/len(all_data['game'])*100:.1f}%)")

   # API error check
   print("\nAPI limits:")
   if 'errors' in result:
       print("API Errors:", result['errors'])
   else:
       print("No API errors found")

   # Missing value checks
   print("\nMissing values in key fields:")
   missing_home_points = sum(1 for g in all_data['game'] if g['homePoints'] is None)
   missing_away_points = sum(1 for g in all_data['game'] if g['awayPoints'] is None)
   missing_start_date = sum(1 for g in all_data['game'] if g['startDate'] is None)

   print(f"Missing home points: {missing_home_points}")
   print(f"Missing away points: {missing_away_points}")
   print(f"Missing start dates: {missing_start_date}")

   # Store validation info
   all_data['validation'] = validation_info

   return all_data

# Execute with proper error handling
try:
   result = asyncio.run(fetch_all_games())
   print("\nData fetch completed successfully!")
except Exception as e:
   print(f"Fatal error during data fetch: {str(e)}")
   raise

# %% [markdown]
# # Data Refinement

# %%
team_info_cols = [
    'teamId', 'year', 'school', 'conference', 'classification', 
    'division', 'talent', 'conferenceId'  
]

game_info_cols = [
    'id',  
    'homeTeam', 'awayTeam', 'homeTeamId', 'awayTeamId', 
    'homePoints', 'awayPoints', 'week', 'season', 'seasonType',
    'startDate', 'neutralSite', 'attendance', 'excitement',
    'homeConference', 'awayConference', 
    'homeStartElo', 'homeEndElo', 'awayStartElo', 'awayEndElo',
    'homePostgameWinProb', 'awayPostgameWinProb',
    'homeLineScores', 'awayLineScores', 
    'conferenceGame'  
]

offense_cols = [
    'epa', 'explosiveness', 
    'passingEpa', 'rushingEpa', 
    'success', 'passingDownsSuccess', 'standardDownsSuccess', 
    'lineYards', 'openFieldYards', 'secondLevelYards'
]

defense_cols = [
    'epaAllowed', 'explosivenessAllowed', 
    'passingEpaAllowed', 'rushingEpaAllowed', 
    'successAllowed', 'passingDownsSuccessAllowed', 'standardDownsSuccessAllowed', 
    'lineYardsAllowed', 'openFieldYardsAllowed', 'secondLevelYardsAllowed'
]

betting_cols = [
    'spread', 'spreadOpen',
    'overUnder', 'overUnderOpen',
    'moneylineHome', 'moneylineAway'
]

weather_cols = [
    'temperature', 'humidity', 'windSpeed', 'windDirection',
    'precipitation', 'pressure', 'snowfall', 'totalSun',
    'weatherConditionCode', 'windGust'
]

ratings_cols = [
    'elo', 'fpi', 'fpiAvgWinProbabilityRank', 'fpiDefensiveEfficiency',
    'fpiGameControlRank', 'fpiOffensiveEfficiency', 'fpiOverallEfficiency',
    'fpiRemainingSosRank', 'fpiResumeRank', 'fpiSosRank', 
    'fpiSpecialTeamsEfficiency', 'fpiStrengthOfRecordRank',
    'spDefense', 'spOffense', 'spOverall', 'spSpecialTeams', 'srs'
]

predicted_points_cols = [
    'distance',
    'down',
    'predictedPoints'
]

matchup_stats_cols = [
    'conferenceGame',
    'homeStartElo', 'awayStartElo',
    'homeEndElo', 'awayEndElo',
    'homePostgameWinProb', 'awayPostgameWinProb'
]

def handle_missing_data(df):
    """Enhanced missing data handling during preprocessing"""
    df = df.copy()
    
    # Base metrics
    scoring_cols = ['homePoints', 'awayPoints']
    for col in scoring_cols:
        df[col] = df[col].fillna(df[col].mean())
    
    # Conference data
    conf_cols = ['home_conference', 'away_conference']
    for col in conf_cols:
        df[col] = df[col].fillna('Unknown')
    
    # Team metrics
    team_metric_cols = [col for col in df.columns if any(x in col for x in ['epa', 'success', 'explosiveness'])]
    for col in team_metric_cols:
        # Fill with conference average first, then overall average
        df[col] = df.groupby('home_conference')[col].transform(lambda x: x.fillna(x.mean()))
        df[col] = df[col].fillna(df[col].mean())
    
    # Weather data
    weather_cols = ['temperature', 'humidity', 'windSpeed']
    for col in weather_cols:
        # Fill with seasonal averages
        df[col] = df.groupby(['season', 'week'])[col].transform(lambda x: x.fillna(x.mean()))
        df[col] = df[col].fillna(df[col].mean())
    
    # Rankings and ratings
    rating_cols = ['elo', 'fpi', 'spOverall', 'srs']
    for col in rating_cols:
        if col in df.columns:
            # Fill with team's previous rating first, then conference average
            df[col] = df.groupby('homeTeamId')[col].transform(lambda x: x.fillna(method='ffill'))
            df[col] = df.groupby('home_conference')[col].transform(lambda x: x.fillna(x.mean()))
            df[col] = df[col].fillna(df[col].mean())
    
    # Rest days and game context
    context_cols = ['rest_days', 'attendance']
    for col in context_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    
    # Default values for missing columns
    required_cols = [
        'epa', 'success', 'explosiveness',
        'home_epa', 'away_epa', 'home_epaAllowed', 'away_epaAllowed',
        'fpi', 'spOverall'
    ]
    
    for col in required_cols:
        if col not in df.columns:
            df[col] = 0
            
    return df

def preprocess_data(df):
    """Main data preprocessing function"""
    print("Starting data preprocessing...")
    
    # Initial data cleaning
    df = df.copy()
    df['startDate'] = pd.to_datetime(df['startDate'])
    
    # Handle missing data
    print("Handling missing data...")
    df = handle_missing_data(df)
    
    # Add derived features
    print("Adding derived features...")
    df = add_enhanced_features(df)
    
    # Convert all numeric columns to float64
    numeric_cols = df.select_dtypes(include=['float32', 'float64', 'int32', 'int64']).columns
    for col in numeric_cols:
        df[col] = df[col].astype(float)
    
    # Data validation
    print("Validating processed data...")
    validation_results = validate_processed_data(df)
    
    if validation_results['has_issues']:
        print("\nWarning: Data quality issues found:")
        for issue in validation_results['issues']:
            print(f"- {issue}")
    
    return df

def validate_processed_data(df):
    """Validate the processed data for common issues"""
    issues = []
    
    # Check for remaining missing values
    missing_cols = df.columns[df.isnull().any()].tolist()
    if missing_cols:
        issues.append(f"Remaining missing values in columns: {missing_cols}")
    
    # Check for infinite values
    inf_cols = df.columns[(~df.isna()) & (df.abs() == np.inf).any()].tolist()
    if inf_cols:
        issues.append(f"Infinite values found in columns: {inf_cols}")
    
    # Validate score ranges
    score_cols = ['homePoints', 'awayPoints']
    for col in score_cols:
        if (df[col] < 0).any() or (df[col] > 100).any():
            issues.append(f"Unrealistic scores found in {col}")
    
    # Validate derived metrics
    if 'home_win_pct' in df.columns and ((df['home_win_pct'] < 0) | (df['home_win_pct'] > 1)).any():
        issues.append("Invalid win percentages found")
    
    return {
        'has_issues': len(issues) > 0,
        'issues': issues
    }

def add_rolling_stats(df, metrics_df, window=5):
    """Add rolling averages for team performance metrics"""
    
    metrics_copy = metrics_df.copy()
    
    # Add offensive metrics
    print(f"\nAdding rolling averages for {len(offense_cols)} offensive metrics...")
    for metric in offense_cols:
        if metric in metrics_copy.columns:
            # Calculate rolling stats
            rolling_stats = (
                metrics_copy.groupby(['teamId', 'year'])[metric]
                .rolling(window=window, min_periods=1)
                .mean()
                .reset_index()
            )
            rolling_stats = rolling_stats.rename(columns={metric: f'{metric}_rolling'})
            
            for team_type in ['home', 'away']:
                try:
                    team_merge = df.merge(
                        rolling_stats,
                        left_on=[f'{team_type}TeamId', 'season'],
                        right_on=['teamId', 'year'],
                        how='left'
                    )
                    df[f'{team_type}_{metric}_rolling_{window}'] = team_merge[f'{metric}_rolling']
                    print(f"Added {team_type}_{metric}_rolling_{window}")
                except Exception as e:
                    print(f"Error adding {metric} for {team_type}: {str(e)}")
    
    # Add defensive metrics
    print(f"\nAdding rolling averages for {len(defense_cols)} defensive metrics...")
    for metric in defense_cols:
        if metric in metrics_copy.columns:
            rolling_stats = (
                metrics_copy.groupby(['teamId', 'year'])[metric]
                .rolling(window=window, min_periods=1)
                .mean()
                .reset_index()
            )
            rolling_stats = rolling_stats.rename(columns={metric: f'{metric}_rolling'})
            
            for team_type in ['home', 'away']:
                try:
                    team_merge = df.merge(
                        rolling_stats,
                        left_on=[f'{team_type}TeamId', 'season'],
                        right_on=['teamId', 'year'],
                        how='left'
                    )
                    df[f'{team_type}_{metric}_rolling_{window}'] = team_merge[f'{metric}_rolling']
                    print(f"Added {team_type}_{metric}_rolling_{window}")
                except Exception as e:
                    print(f"Error adding {metric} for {team_type}: {str(e)}")


    # Add ratings rolling averages
    print(f"\nAdding rolling averages for {len(ratings_cols)} rating metrics...")
    for metric in ratings_cols:
        if metric in metrics_copy.columns:
            rolling_stats = (
                metrics_copy.groupby(['teamId', 'year'])[metric]
                .rolling(window=window, min_periods=1)
                .mean()
                .reset_index()
            )
            rolling_stats = rolling_stats.rename(columns={metric: f'{metric}_rolling'})
            
            for team_type in ['home', 'away']:
                try:
                    team_merge = df.merge(
                        rolling_stats,
                        left_on=[f'{team_type}TeamId', 'season'],
                        right_on=['teamId', 'year'],
                        how='left'
                    )
                    df[f'{team_type}_{metric}_rolling_{window}'] = team_merge[f'{metric}_rolling']
                    print(f"Added {team_type}_{metric}_rolling_{window}")
                except Exception as e:
                    print(f"Error adding {metric} for {team_type}: {str(e)}")
    
    return df

# Add h2h stats
def add_head_to_head_stats(df, lookback_years=3):
    """Add historical head-to-head performance metrics"""
    df = df.sort_values('startDate')
    
    def get_h2h_stats(row):
        mask = (
            ((df['homeTeamId'] == row['homeTeamId']) & 
             (df['awayTeamId'] == row['awayTeamId']) |
             (df['homeTeamId'] == row['awayTeamId']) & 
             (df['awayTeamId'] == row['homeTeamId'])) &
            (df['startDate'] < row['startDate']) &
            (df['season'] >= row['season'] - lookback_years)
        )
        
        h2h_games = df[mask]
        
        return pd.Series({
            'h2h_games': len(h2h_games),
            'h2h_home_wins': sum((h2h_games['homeTeamId'] == row['homeTeamId']) & 
                               (h2h_games['homePoints'] > h2h_games['awayPoints'])),
            'h2h_avg_margin': (h2h_games['homePoints'] - h2h_games['awayPoints']).mean()
        })
    
    h2h_stats = df.apply(get_h2h_stats, axis=1)
    df = pd.concat([df, h2h_stats], axis=1)
    
    return df

# Add days between games
def add_game_context(df):
    """Add contextual features like rest days"""
    df = df.sort_values(['startDate', 'homeTeamId'])
    
    for team_type in ['home', 'away']:
        team_id_col = f'{team_type}TeamId'
        
        if not pd.api.types.is_datetime64_any_dtype(df['startDate']):
            df['startDate'] = pd.to_datetime(df['startDate'])
        
        df[f'{team_type}_rest_days'] = df.groupby(team_id_col)['startDate'].diff().dt.days
        
    return df

# Create win / loss streaks
def add_streaks(df):
    """Add team winning/losing streaks"""
    df = df.sort_values('startDate')
    
    for team_type in ['home', 'away']:
        team_id_col = f'{team_type}TeamId'
        points_col = f'{team_type}Points'
        
        # Create win/loss indicator
        df[f'{team_type}_won'] = (
            df[points_col] > df[f'{"away" if team_type == "home" else "home"}Points']
        ).astype(int)
        
        # Calculate win streaks
        df[f'{team_type}_win_streak'] = df.groupby([team_id_col, 'season'])[f'{team_type}_won'].transform(
            lambda x: x * (x.groupby((x != x.shift()).cumsum()).cumcount() + 1)
        ).fillna(0)
        
        # Calculate loss streaks - fix the error by creating a loss indicator first
        df[f'{team_type}_lost'] = (~df[f'{team_type}_won'].astype(bool)).astype(int)
        df[f'{team_type}_loss_streak'] = df.groupby([team_id_col, 'season'])[f'{team_type}_lost'].transform(
            lambda x: x * (x.groupby((x != x.shift()).cumsum()).cumcount() + 1)
        ).fillna(0)
        
        # Clean up intermediate columns
        df = df.drop(columns=[f'{team_type}_lost'])
        
        # Add combined streak (positive for wins, negative for losses)
        df[f'{team_type}_streak'] = df[f'{team_type}_win_streak'] - df[f'{team_type}_loss_streak']
    
    return df

def add_quality_metrics(df):
    """Add data quality flags and confidence scores"""
    df['data_quality_score'] = 100
    
    # Reduce score for missing critical data
    critical_cols = ['homePoints', 'awayPoints', 'spread', 'overUnder']
    for col in critical_cols:
        df.loc[df[col].isnull(), 'data_quality_score'] -= 25
        
    # Flag games with suspicious data
    df['suspicious_score'] = (
        (df['homePoints'] > 100) | 
        (df['awayPoints'] > 100) |
        (abs(df['spread']) > 50)
    )
    return df


# Create DataFrames from API result
print("Creating DataFrames from API result...")
metrics_df = pd.DataFrame(result['adjustedTeamMetrics'])
teams_df = pd.DataFrame(result['currentTeams'])
games_df = pd.DataFrame(result['game'])
lines_df = pd.DataFrame(result['gameLines'])
weather_df = pd.DataFrame(result['gameWeather'])
talent_df = pd.DataFrame(result['teamTalent'])

# Let's check if ratings and predictedPoints are available before creating their DataFrames
if 'ratings' in result:
    ratings_df = pd.DataFrame(result['ratings'])
else:
    print("Note: ratings data not available in API response")
    ratings_df = pd.DataFrame()

if 'predictedPoints' in result:
    predicted_points_df = pd.DataFrame(result['predictedPoints'])
else:
    print("Note: predicted points data not available in API response")
    predicted_points_df = pd.DataFrame()

# Clean and convert numeric columns in lines_df
numeric_columns = ['spread', 'spreadOpen', 'overUnder', 'overUnderOpen', 
                  'moneylineHome', 'moneylineAway']
for col in numeric_columns:
    if col in lines_df.columns:
        lines_df[col] = pd.to_numeric(lines_df[col], errors='coerce')

# Process nested team data in metrics
if 'team' in metrics_df.columns:
    team_info = pd.json_normalize(metrics_df['team'].dropna())
    metrics_df = metrics_df.drop('team', axis=1)
    for col in team_info.columns:
        metrics_df[f'team_{col}'] = team_info[col]

# Process nested team info in games
if 'homeTeamInfo' in games_df.columns:
    home_info = pd.json_normalize(games_df['homeTeamInfo'].dropna())
    games_df = games_df.drop('homeTeamInfo', axis=1)
    for col in home_info.columns:
        games_df[f'home_{col}'] = home_info[col].fillna('Unknown')

if 'awayTeamInfo' in games_df.columns:
    away_info = pd.json_normalize(games_df['awayTeamInfo'].dropna())
    games_df = games_df.drop('awayTeamInfo', axis=1)
    for col in away_info.columns:
        games_df[f'away_{col}'] = away_info[col].fillna('Unknown')

# Process nested team data in talent
if 'team' in talent_df.columns:
    talent_team_info = pd.json_normalize(talent_df['team'].dropna())
    talent_df = talent_df.drop('team', axis=1)
    for col in talent_team_info.columns:
        talent_df[f'team_{col}'] = talent_team_info[col]

# Merge all data with games
print("\nMerging data sources...")
games_df = pd.merge(games_df, lines_df, left_on='id', right_on='gameId', how='left')
games_df = pd.merge(games_df, weather_df, left_on='id', right_on='gameId', how='left')
games_df = pd.merge(games_df, ratings_df, left_on=['homeTeamId', 'season'], 
                   right_on=['teamId', 'year'], how='left', suffixes=('', '_home'))
games_df = pd.merge(games_df, ratings_df, left_on=['awayTeamId', 'season'], 
                   right_on=['teamId', 'year'], how='left', suffixes=('', '_away'))

# Convert numeric columns to appropriate types
numeric_game_cols = ['homePoints', 'awayPoints', 'homeStartElo', 'homeEndElo', 
                    'awayStartElo', 'awayEndElo', 'homePostgameWinProb', 
                    'awayPostgameWinProb', 'temperature', 'humidity', 'windSpeed'] + \
                    ratings_cols
for col in numeric_game_cols:
    if col in games_df.columns:
        games_df[col] = pd.to_numeric(games_df[col], errors='coerce')

# Add new features
print("\nAdding advanced features...")
games_df = add_rolling_stats(games_df, metrics_df)
games_df = add_head_to_head_stats(games_df)
games_df = add_game_context(games_df)
games_df = add_streaks(games_df)

# Print comprehensive data quality report
print("\nData Overview:")
print("-" * 50)
print(f"Total unique games: {len(games_df)}")
print(f"Years in data: {sorted(games_df['season'].unique())}")
print(f"Teams: {games_df['homeTeamId'].nunique()}")

# Print conference information
home_conferences = games_df['home_conference'].dropna().unique()
away_conferences = games_df['away_conference'].dropna().unique() 
all_conferences = sorted(set(home_conferences) | set(away_conferences))
print(f"Conferences: {all_conferences}")

# Print games per year distribution
print("\nGames per year:")
games_by_year = games_df.groupby('season').size()
print(games_by_year)

# Print metrics sample
print("\nSample metrics:")
all_metrics = offense_cols + defense_cols + ratings_cols
metrics_to_show = [col for col in all_metrics if col in games_df.columns]
for metric in metrics_to_show:
    mean_val = pd.to_numeric(games_df[metric], errors='coerce').mean()
    std_val = pd.to_numeric(games_df[metric], errors='coerce').std()
    print(f"{metric}:")
    print(f"  Mean: {mean_val:.3f}")
    print(f"  Std:  {std_val:.3f}")

# Print data completeness by category
print("\nData Completeness by Category:")
for category, cols in [
    ("Team & Game Info", team_info_cols + game_info_cols),
    ("Offensive Metrics", offense_cols),
    ("Defensive Metrics", defense_cols),
    ("Betting Lines", betting_cols),
    ("Weather Data", weather_cols),
    ("Matchup Stats", matchup_stats_cols),
    ("Ratings", ratings_cols),
    ("Predicted Points", predicted_points_cols)
]:
    print(f"\n{category}:")
    for col in cols:
        if col in games_df.columns:
            pct = (1 - games_df[col].isnull().mean()) * 100
            print(f"{col}: {pct:.1f}% complete")

# Print seasonal breakdown
print("\nSeason Breakdown:")
print("{:<8} | {:<7} | {:<14} | {:<16}".format("Season", "Games", "Games w/Lines", "Games w/Weather"))
print("-" * 50)

season_stats = games_df.groupby('season').agg({
    'homePoints': 'count',
    'spread': lambda x: pd.to_numeric(x, errors='coerce').notna().sum(),
    'temperature': lambda x: pd.to_numeric(x, errors='coerce').notna().sum()
}).fillna(0).astype(int)

for season in sorted(games_df['season'].unique()):
    if season in season_stats.index:
        row = season_stats.loc[season]
        print("{:<8} | {:<7} | {:<14} | {:<16}".format(
            season, row['homePoints'], row['spread'], row['temperature']
        ))

# Generate summary statistics
print("\nSummary Statistics:")
for stat_type in ['home', 'away']:
    print(f"\n{stat_type.title()} Team Statistics:")
    points = pd.to_numeric(games_df[f'{stat_type}Points'], errors='coerce')
    print(f"Average points: {points.mean():.1f}")
    print(f"Point standard deviation: {points.std():.1f}")

if 'spread' in games_df.columns:
    spread = pd.to_numeric(games_df['spread'], errors='coerce')
    over_under = pd.to_numeric(games_df['overUnder'], errors='coerce')
    print(f"\nBetting Statistics:")
    print(f"Average spread: {spread.mean():.1f}")
    print(f"Average over/under: {over_under.mean():.1f}")

print("\nData processing completed successfully!")

# Create visualization of data completeness
plt.figure(figsize=(12, 6))
plt.bar(season_stats.index, season_stats['homePoints'], label='Total Games')
plt.bar(season_stats.index, season_stats['spread'], label='Games with Lines')
plt.bar(season_stats.index, season_stats['temperature'], label='Games with Weather')
plt.title('Data Completeness by Season')
plt.xlabel('Season')
plt.ylabel('Number of Games')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print(games_df)

# %%
# Check if data is loaded
print("Games DataFrame loaded:", games_df is not None)
print("Games DataFrame shape:", games_df.shape if games_df is not None else "Not loaded")

# %% [markdown]
# # Data Visualization

# %%
# Split home and away metrics for better analysis
print("\nSample game with detailed metrics:")
metric_cols = ['season', 'week', 'homeTeam', 'awayTeam', 'homePoints', 'awayPoints',
              'spread', 'overUnder', 'temperature',
              'epa', 'epaAllowed', 'success', 'successAllowed',
              'elo', 'fpi', 'spOverall', 'srs',  # Added rating metrics
              'fpiOffensiveEfficiency', 'fpiDefensiveEfficiency', 'fpiSpecialTeamsEfficiency']
existing_cols = [col for col in metric_cols if col in games_df.columns]
game_view = games_df[existing_cols].head(1)
print(game_view.to_string())

# Enhanced basic stats
print("\nDetailed Game Stats:")
home_points = pd.to_numeric(games_df['homePoints'], errors='coerce')
away_points = pd.to_numeric(games_df['awayPoints'], errors='coerce')
total_points = home_points + away_points

print(f"Home Points - Mean: {home_points.mean():.1f}, Std: {home_points.std():.1f}")
print(f"Away Points - Mean: {away_points.mean():.1f}, Std: {away_points.std():.1f}")
print(f"Total Points - Mean: {total_points.mean():.1f}, Std: {total_points.std():.1f}")
print(f"Home Win %: {(home_points > away_points).mean()*100:.1f}%")

# Add Rating Stats
print("\nRating Stats:")
rating_cols = ['elo', 'fpi', 'spOverall', 'srs']
for col in rating_cols:
    if col in games_df.columns:
        values = pd.to_numeric(games_df[col], errors='coerce')
        print(f"{col} - Mean: {values.mean():.1f}, Std: {values.std():.1f}")

# Efficiency Stats
print("\nEfficiency Stats:")
efficiency_cols = ['fpiOffensiveEfficiency', 'fpiDefensiveEfficiency', 'fpiSpecialTeamsEfficiency']
for col in efficiency_cols:
    if col in games_df.columns:
        values = pd.to_numeric(games_df[col], errors='coerce')
        print(f"{col} - Mean: {values.mean():.1f}, Std: {values.std():.1f}")

# Betting line analysis with enhanced visualization
if 'spread' in games_df.columns and 'overUnder' in games_df.columns:
    spread = pd.to_numeric(games_df['spread'], errors='coerce')
    over_under = pd.to_numeric(games_df['overUnder'], errors='coerce')
    
    print("\nBetting Line Stats:")
    print(f"Spread - Mean: {spread.mean():.1f}, Std: {spread.std():.1f}")
    print(f"Over/Under - Mean: {over_under.mean():.1f}, Std: {over_under.std():.1f}")
    
    # Create a figure with 2x2 subplots for enhanced visualization
    plt.figure(figsize=(15, 10))
    
    # Spread distribution
    plt.subplot(2, 2, 1)
    sns.histplot(spread.dropna(), bins=30)
    plt.title('Spread Distribution')
    
    # Over/Under distribution
    plt.subplot(2, 2, 2)
    sns.histplot(over_under.dropna(), bins=30)
    plt.title('Over/Under Distribution')
    
    # Rating distributions
    plt.subplot(2, 2, 3)
    if 'elo' in games_df.columns:
        sns.histplot(pd.to_numeric(games_df['elo'], errors='coerce').dropna(), bins=30)
        plt.title('ELO Rating Distribution')
    
    plt.subplot(2, 2, 4)
    if 'fpi' in games_df.columns:
        sns.histplot(pd.to_numeric(games_df['fpi'], errors='coerce').dropna(), bins=30)
        plt.title('FPI Distribution')
    
    plt.tight_layout()
    plt.show()

# Enhanced data completeness check
print("\nDetailed Data Completeness:")
important_cols = ['homePoints', 'awayPoints', 'spread', 'overUnder', 
                 'epa', 'success', 'elo', 'fpi', 'spOverall', 'srs',
                 'fpiOffensiveEfficiency', 'fpiDefensiveEfficiency', 'fpiSpecialTeamsEfficiency']
completeness_data = []
for col in important_cols:
    if col in games_df.columns:
        valid_data = pd.to_numeric(games_df[col], errors='coerce').notna()
        pct_complete = valid_data.mean() * 100
        completeness_data.append({
            'column': col,
            'completeness': pct_complete,
            'missing_count': (~valid_data).sum()
        })

completeness_df = pd.DataFrame(completeness_data)
print("\nCompleteness Summary:")
print(completeness_df.to_string(index=False))

# Original data source validation with ratings
print("\nData Source Validation:")
games_by_season = games_df.groupby('season').agg({
    'homePoints': 'count',
    'spread': lambda x: pd.to_numeric(x, errors='coerce').notna().sum(),
    'temperature': lambda x: pd.to_numeric(x, errors='coerce').notna().sum(),
    'elo': lambda x: pd.to_numeric(x, errors='coerce').notna().sum(),
    'fpi': lambda x: pd.to_numeric(x, errors='coerce').notna().sum()
}).fillna(0).astype(int)

print("\nSeason Breakdown:")
print("Season | Games | Lines | Weather | ELO | FPI")
print("-" * 60)
for season in sorted(games_df['season'].unique()):
    if season in games_by_season.index:
        row = games_by_season.loc[season]
        print(f"{season:6d} | {row['homePoints']:5d} | {row['spread']:5d} | {row['temperature']:7d} | {row['elo']:4d} | {row['fpi']:4d}")

# Check data types and memory usage
print("\nDetailed Column Information:")
print(games_df[existing_cols].info(memory_usage='deep'))

# Save any problematic rows for investigation
problematic_rows = games_df[
    (pd.to_numeric(games_df['homePoints'], errors='coerce').isna()) |
    (pd.to_numeric(games_df['awayPoints'], errors='coerce').isna()) |
    (pd.to_numeric(games_df['elo'], errors='coerce').isna() & 
     pd.to_numeric(games_df['fpi'], errors='coerce').isna())
]
if len(problematic_rows) > 0:
    print("\nProblematic Rows Found:")
    print(problematic_rows[existing_cols].head())


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List

def plot_score_distributions(games_df: pd.DataFrame) -> None:
    """Create score distribution plots by conference."""
    plt.figure(figsize=(15, 10))
    
    # Create 2x2 subplots
    plt.subplot(2, 2, 1)
    sns.kdeplot(data=games_df, x='homePoints', hue='home_conference')
    plt.title('Home Points Distribution by Conference')
    
    plt.subplot(2, 2, 2)
    sns.kdeplot(data=games_df, x='awayPoints', hue='away_conference')
    plt.title('Away Points Distribution by Conference')
    
    plt.subplot(2, 2, 3)
    total_points = games_df['homePoints'] + games_df['awayPoints']
    sns.histplot(total_points, bins=30)
    plt.title('Total Points Distribution')
    
    plt.subplot(2, 2, 4)
    point_diff = games_df['homePoints'] - games_df['awayPoints']
    sns.histplot(point_diff, bins=30)
    plt.title('Point Differential Distribution')
    
    plt.tight_layout()
    plt.show()

def plot_correlation_matrix(games_df: pd.DataFrame) -> None:
    """Create correlation heatmap for key features."""
    correlation_features = [
        'homePoints', 'awayPoints', 'spread', 'overUnder',
        'homeStartElo', 'awayStartElo', 
        'home_epa_rolling_5', 'away_epa_rolling_5',
        'temperature', 'home_success_rolling_5', 'away_success_rolling_5'
    ]
    
    # Filter for existing columns
    existing_features = [col for col in correlation_features if col in games_df.columns]
    correlation_df = games_df[existing_features].corr()
    
    plt.figure(figsize=(12, 8))
    mask = np.triu(np.ones_like(correlation_df), k=1)
    sns.heatmap(correlation_df, annot=True, cmap='coolwarm', center=0, 
                mask=mask, fmt='.2f', square=True)
    plt.title('Feature Correlations')
    plt.tight_layout()
    plt.show()

def plot_time_series_analysis(games_df: pd.DataFrame) -> None:
    """Create time series visualizations."""
    # Convert date and create time-based aggregations
    games_df['date'] = pd.to_datetime(games_df['startDate'])
    weekly_scores = games_df.groupby([
        games_df['date'].dt.isocalendar().year,
        games_df['date'].dt.isocalendar().week
    ]).agg({
        'homePoints': 'mean',
        'awayPoints': 'mean',
        'spread': 'mean',
        'overUnder': 'mean'
    }).reset_index()
    
    plt.figure(figsize=(15, 10))
    
    # Scoring trends
    plt.subplot(2, 1, 1)
    plt.plot(weekly_scores.index, weekly_scores['homePoints'], 
             label='Home Points', alpha=0.7)
    plt.plot(weekly_scores.index, weekly_scores['awayPoints'], 
             label='Away Points', alpha=0.7)
    plt.title('Weekly Scoring Trends')
    plt.xlabel('Week Number')
    plt.ylabel('Average Points')
    plt.legend()
    
    # Betting line trends
    plt.subplot(2, 1, 2)
    plt.plot(weekly_scores.index, weekly_scores['spread'], 
             label='Average Spread', alpha=0.7)
    plt.plot(weekly_scores.index, weekly_scores['overUnder'], 
             label='Average Over/Under', alpha=0.7)
    plt.title('Weekly Betting Line Trends')
    plt.xlabel('Week Number')
    plt.ylabel('Points')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def plot_conference_performance(games_df: pd.DataFrame) -> None:
    """Analyze and visualize conference performance."""
    plt.figure(figsize=(15, 10))
    
    # Conference scoring
    plt.subplot(2, 1, 1)
    conf_stats = games_df.groupby('home_conference').agg({
        'homePoints': ['mean', 'std']
    }).round(2)
    
    conf_means = conf_stats[('homePoints', 'mean')]
    conf_stds = conf_stats[('homePoints', 'std')]
    
    x = range(len(conf_means))
    plt.bar(x, conf_means, yerr=conf_stds, capsize=5)
    plt.xticks(x, conf_means.index, rotation=45)
    plt.title('Average Points by Conference')
    plt.ylabel('Points')
    
    # Conference win rates
    plt.subplot(2, 1, 2)
    games_df['home_win'] = (games_df['homePoints'] > games_df['awayPoints']).astype(int)
    win_rates = games_df.groupby('home_conference')['home_win'].mean().sort_values(ascending=False)
    
    win_rates.plot(kind='bar')
    plt.title('Home Win Rate by Conference')
    plt.ylabel('Win Rate')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()

def plot_weather_impact(games_df: pd.DataFrame) -> None:
    """Analyze and visualize weather impacts on scoring."""
    plt.figure(figsize=(15, 10))
    
    # Temperature impact
    plt.subplot(2, 2, 1)
    sns.scatterplot(data=games_df, x='temperature', 
                    y='homePoints', alpha=0.5)
    plt.title('Temperature vs Home Points')
    
    plt.subplot(2, 2, 2)
    total_points = games_df['homePoints'] + games_df['awayPoints']
    sns.scatterplot(data=games_df, x='temperature', 
                    y=total_points, alpha=0.5)
    plt.title('Temperature vs Total Points')
    
    # Wind impact
    plt.subplot(2, 2, 3)
    sns.scatterplot(data=games_df, x='windSpeed', 
                    y='homePoints', alpha=0.5)
    plt.title('Wind Speed vs Home Points')
    
    # Temperature bands analysis
    plt.subplot(2, 2, 4)
    games_df['temp_band'] = pd.qcut(games_df['temperature'], q=5)
    sns.boxplot(data=games_df, x='temp_band', y='homePoints')
    plt.title('Points Distribution by Temperature Band')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()

def plot_betting_analysis(games_df: pd.DataFrame) -> None:
    """Analyze and visualize betting line accuracy."""
    plt.figure(figsize=(15, 10))
    
    # Create betting-related features
    games_df['actual_margin'] = games_df['homePoints'] - games_df['awayPoints']
    games_df['spread_error'] = games_df['actual_margin'] - games_df['spread']
    games_df['total_points'] = games_df['homePoints'] + games_df['awayPoints']
    games_df['ou_error'] = games_df['total_points'] - games_df['overUnder']
    
    # Spread accuracy
    plt.subplot(2, 2, 1)
    sns.histplot(games_df['spread_error'], bins=30)
    plt.title('Spread Prediction Error Distribution')
    
    # Over/Under accuracy
    plt.subplot(2, 2, 2)
    sns.histplot(games_df['ou_error'], bins=30)
    plt.title('Over/Under Prediction Error Distribution')
    
    # Spread vs Actual
    plt.subplot(2, 2, 3)
    sns.scatterplot(data=games_df, x='spread', y='actual_margin', alpha=0.5)
    plt.axline([0, 0], [1, 1], color='r', linestyle='--')
    plt.title('Predicted vs Actual Margin')
    
    # Over/Under vs Actual
    plt.subplot(2, 2, 4)
    sns.scatterplot(data=games_df, x='overUnder', y='total_points', alpha=0.5)
    plt.axline([0, 0], [1, 1], color='r', linestyle='--')
    plt.title('Predicted vs Actual Total Points')
    
    plt.tight_layout()
    plt.show()

def create_all_visualizations(games_df: pd.DataFrame) -> None:
    """Create all visualizations in sequence."""
    plot_score_distributions(games_df)
    plot_correlation_matrix(games_df)
    plot_time_series_analysis(games_df)
    plot_conference_performance(games_df)
    plot_weather_impact(games_df)
    plot_betting_analysis(games_df)

# Usage example:
# create_all_visualizations(games_d

# %%
# Print metrics columns to see what we have
print("Available metrics columns:")
print(sorted(metrics_df.columns.tolist()))  # Sorted for easier reading

# Check both metrics_df and games_df for offensive metrics
print("\nChecking offensive metrics availability:")
metrics_cols_present = {col: col in metrics_df.columns for col in offense_cols}
games_cols_present = {col: col in games_df.columns for col in offense_cols}

print("\nIn metrics_df:")
for col, present in metrics_cols_present.items():
    print(f"{col}: {present}")

print("\nIn games_df:")
for col, present in games_cols_present.items():
    print(f"{col}: {present}")

# Verify column data types and non-null counts
print("\nMetrics columns info:")
print(metrics_df[metrics_df.columns].info())

# %% [markdown]
# # Data Quality Checks

# %%
# Data quality checks after processing
print("\nData Quality Check:")
print("-" * 50)

# Check for key metrics completeness
key_metrics = ['epa', 'success', 'explosiveness', 'talent', 'elo', 'fpi', 'sp', 'srs']
print("\nKey Metrics Completeness:")
for metric in key_metrics:
    metric_cols = [col for col in games_df.columns if metric in col.lower()]
    for col in metric_cols:
        completeness = (1 - games_df[col].isnull().mean()) * 100
        print(f"{col}: {completeness:.1f}% complete")

# Verify temporal coverage
print("\nTemporal Coverage Analysis:")
agg_dict = {
    'id': 'count',
    'homePoints': ['mean', 'std'],
    'elo': ['mean', 'std'],
    'fpi': ['mean', 'std'],
    'srs': ['mean', 'std']
}

games_by_year = games_df.groupby('season').agg(agg_dict).round(2)
print(games_by_year)

# Check for potential duplicates
potential_dupes = games_df.groupby(['homeTeam', 'awayTeam', 'week', 'season']).size()
if (potential_dupes > 1).any():
    print("\nWarning: Found potential duplicate games!")
    print(potential_dupes[potential_dupes > 1])

# Verify feature engineering results
print("\nFeature Engineering Verification:")
rolling_features = [col for col in games_df.columns if 'rolling' in col]
h2h_features = [col for col in games_df.columns if 'h2h' in col]
streak_features = [col for col in games_df.columns if 'streak' in col]
rating_features = [col for col in games_df.columns if any(x in col.lower() for x in ['elo', 'fpi', 'sp', 'srs'])]

print(f"\nTotal rolling features: {len(rolling_features)}")
if rolling_features:
    print("Sample rolling features:", rolling_features[:5])
print(f"\nTotal h2h features: {len(h2h_features)}")
if h2h_features:
    print("H2H features:", h2h_features)
print(f"\nTotal streak features: {len(streak_features)}")
if streak_features:
    print("Streak features:", streak_features)
print(f"\nTotal rating features: {len(rating_features)}")
if rating_features:
    print("Sample rating features:", rating_features[:5])

# Quick correlation analysis with target variables
target_vars = ['homePoints', 'awayPoints']
numeric_cols = games_df.select_dtypes(include=[np.number]).columns

correlations = {}
for target in target_vars:
    if target in games_df.columns:
        correlations[target] = games_df[numeric_cols].corrwith(games_df[target]).sort_values(ascending=False)

print("\nTop correlations with points:")
for target, corrs in correlations.items():
    print(f"\n{target}:")
    print(corrs.head(10))

# Enhanced visualizations
plt.figure(figsize=(15, 10))

# Points distribution
plt.subplot(221)
sns.histplot(data=games_df['homePoints'].dropna(), bins=30, label='Home')
sns.histplot(data=games_df['awayPoints'].dropna(), bins=30, label='Away', alpha=0.6)
plt.title('Points Distribution')
plt.legend()

# Ratings distributions
plt.subplot(222)
if 'elo' in games_df.columns:
    sns.histplot(data=games_df['elo'].dropna(), bins=30, label='ELO')
if 'fpi' in games_df.columns:
    sns.histplot(data=games_df['fpi'].dropna(), bins=30, label='FPI', alpha=0.6)
plt.title('Ratings Distribution')
plt.legend()

# Points by year
plt.subplot(223)
sns.boxplot(data=games_df, x='season', y='homePoints')
plt.title('Home Points Distribution by Year')
plt.xticks(rotation=45)

# Ratings by year
plt.subplot(224)
if 'elo' in games_df.columns:
    sns.boxplot(data=games_df, x='season', y='elo')
    plt.title('ELO Rating Distribution by Year')
    plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# Memory usage analysis
memory_usage = games_df.memory_usage(deep=True) / 1024**2  # Convert to MB
print("\nMemory Usage Analysis:")
print(f"Total memory usage: {memory_usage.sum():.2f} MB")
print("\nLargest columns by memory usage:")
print(memory_usage.sort_values(ascending=False).head())

# Enhanced summary statistics with ratings
cols_to_summarize = ['homePoints', 'awayPoints', 'elo', 'fpi', 'srs', 
                     'fpiOffensiveEfficiency', 'fpiDefensiveEfficiency', 
                     'fpiSpecialTeamsEfficiency']
summary_stats = games_df[cols_to_summarize].describe()
print("\nSummary Statistics:")
print(summary_stats)

# Additional analysis of rating metrics consistency
print("\nRating Metrics Consistency Check:")
rating_pairs = [('elo', 'fpi'), ('fpi', 'srs'), ('elo', 'srs')]
for rat1, rat2 in rating_pairs:
    if rat1 in games_df.columns and rat2 in games_df.columns:
        correlation = games_df[rat1].corr(games_df[rat2])
        print(f"Correlation between {rat1.upper()} and {rat2.upper()}: {correlation:.3f}")

# Keep your existing imports and initial checks

# Add the new quality check functions first
def check_data_quality(games_df):
    """Run comprehensive data quality checks with handling for unhashable types"""
    quality_report = {}
    
    # Basic metrics (excluding duplicate check for unhashable types)
    quality_report['basic_metrics'] = {
        'total_rows': len(games_df),
        'total_columns': len(games_df.columns)
    }
    
    # Check duplicates only on hashable columns
    hashable_cols = [col for col in games_df.columns 
                    if games_df[col].dtype != 'object' or 
                    not games_df[col].apply(lambda x: isinstance(x, (list, dict))).any()]
    
    quality_report['basic_metrics']['duplicate_rows'] = games_df[hashable_cols].duplicated().sum()
    quality_report['basic_metrics']['memory_usage_mb'] = games_df.memory_usage(deep=True).sum() / 1024**2
    
    # Missing value analysis
    missing_data = pd.DataFrame({
        'missing_count': games_df.isnull().sum(),
        'missing_percentage': (games_df.isnull().sum() / len(games_df) * 100).round(2)
    }).sort_values('missing_percentage', ascending=False)
    quality_report['missing_data'] = missing_data[missing_data['missing_count'] > 0]
    
    # Column type analysis
    quality_report['column_types'] = {
        'numeric': len(games_df.select_dtypes(include=['int64', 'float64']).columns),
        'categorical': len(games_df.select_dtypes(include=['object', 'category']).columns),
        'datetime': len(games_df.select_dtypes(include=['datetime64']).columns),
        'other': len(games_df.select_dtypes(exclude=['int64', 'float64', 'object', 'category', 'datetime64']).columns)
    }
    
    # Check outliers for numeric columns
    numeric_cols = games_df.select_dtypes(include=['int64', 'float64']).columns
    outliers_report = {}
    for col in numeric_cols:
        Q1 = games_df[col].quantile(0.25)
        Q3 = games_df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = len(games_df[(games_df[col] < (Q1 - 1.5 * IQR)) | 
                               (games_df[col] > (Q3 + 1.5 * IQR))])
        
        # Fix: Calculate percentage and round after
        percentage = float(outliers) / len(games_df) * 100
        outliers_report[col] = {
            'outliers_count': outliers,
            'outliers_percentage': round(percentage, 2)
        }
    quality_report['outliers'] = outliers_report
    
    return quality_report

# Usage remains the same:
quality_report = check_data_quality(games_df)

print("\nBasic Metrics:")
print("-" * 50)
for metric, value in quality_report['basic_metrics'].items():
    print(f"{metric}: {value}")

print("\nColumn Types:")
print("-" * 50)
for type_name, count in quality_report['column_types'].items():
    print(f"{type_name}: {count}")

print("\nOutliers Analysis:")
print("-" * 50)
for col, stats in quality_report['outliers'].items():
    if stats['outliers_count'] > 0:
        print(f"{col}: {stats['outliers_count']} outliers ({stats['outliers_percentage']:.2f}%)")

# Then, after your existing quality checks, add:
print("\nRunning Enhanced Quality Checks...")
quality_report = check_data_quality(games_df)

print("\nBasic Metrics:")
print("-" * 50)
for metric, value in quality_report['basic_metrics'].items():
    print(f"{metric}: {value}")

print("\nMissing Data Analysis:")
print("-" * 50)
print(quality_report['missing_data'])

# Enhanced visualizations - add after your existing plots
plt.figure(figsize=(15, 10))

# Missing data heatmap
plt.subplot(221)
sns.heatmap(games_df[games_df.columns[:20]].isnull(), 
            yticklabels=False, 
            cbar=False,
            cmap='viridis')
plt.title('Missing Data Pattern (First 20 Columns)')

# Points vs Ratings scatter
plt.subplot(222)
if 'elo' in games_df.columns and 'homePoints' in games_df.columns:
    sns.scatterplot(data=games_df, x='elo', y='homePoints', alpha=0.5)
plt.title('Points vs ELO Rating')

# Feature completeness
plt.subplot(223)
# Fix: Only use metrics that exist in the dataframe
available_metrics = []
for metric in key_metrics:
    metric_cols = [col for col in games_df.columns if metric in col.lower()]
    available_metrics.extend(metric_cols[:1])  # Take first instance of each metric type

available_metrics = list(set(available_metrics))  # Remove duplicates
if available_metrics:
    completeness = (1 - games_df[available_metrics].isnull().mean()) * 100
    completeness.plot(kind='bar')
    plt.title('Feature Completeness')
    plt.xticks(rotation=45)

# Correlation heatmap
plt.subplot(224)
key_numeric_cols = ['homePoints', 'awayPoints', 'elo', 'fpi', 'srs']
available_numeric_cols = [col for col in key_numeric_cols if col in games_df.columns]
if available_numeric_cols:
    sns.heatmap(games_df[available_numeric_cols].corr(), 
                annot=True, 
                cmap='coolwarm',
                fmt='.2f')
plt.title('Key Metrics Correlation')

plt.tight_layout()
plt.show()

# %% [markdown]
# # Feature Engineering

# %%
def add_team_performance_context(games_df):
   """Add team performance context features"""
   df = games_df.copy()
   
   for team_type in ['home', 'away']:
       team_id_col = f'{team_type}TeamId'
       points_col = f'{team_type}Points'
       
       # Historical scoring
       df[f'{team_type}_historical_scoring'] = df.groupby(team_id_col)[points_col].transform(
           lambda x: x.expanding().mean()
       )
       
       # Last 3/5 game averages
       for window in [3, 5]:
           df[f'{team_type}_last{window}_avg'] = df.groupby(team_id_col)[points_col].transform(
               lambda x: x.rolling(window, min_periods=1).mean()
           )
           
       # Season stats
       df[f'{team_type}_season_avg'] = df.groupby([team_id_col, 'season'])[points_col].transform('mean')
       df[f'{team_type}_season_std'] = df.groupby([team_id_col, 'season'])[points_col].transform('std')
       
       # Win/Loss records
       if f'{team_type}_won' not in df.columns:
           opp_type = 'away' if team_type == 'home' else 'home' 
           df[f'{team_type}_won'] = (df[points_col] > df[f'{opp_type}Points']).astype(int)
           
       # Conference performance
       df[f'{team_type}_conf_win_pct'] = df.groupby(
           ['season', f'{team_type}_conference'])[f'{team_type}_won'].transform('mean')
           
       # Home/Away splits
       df[f'{team_type}_venue_scoring'] = df.groupby([team_id_col, 'season'])[points_col].transform(
           lambda x: x.expanding().mean()
       )
       
   # Conference matchups
   df['conf_vs_conf_history'] = df.groupby(['home_conference', 'away_conference'])['home_won'].transform('mean')
   
   return df

def add_enhanced_timing_features(games_df):
   """Add timing-related features"""
   df = games_df.copy()
   
   # Season context
   df['games_into_season'] = df.groupby(['season', 'homeTeamId']).cumcount()
   df['games_remaining'] = df.groupby('season')['week'].transform('max') - df['week']
   df['season_progress'] = df['week'] / df.groupby('season')['week'].transform('max')
   
   # Rest days
   if all(col in df.columns for col in ['home_rest_days', 'away_rest_days']):
       df['rest_advantage'] = df['home_rest_days'] - df['away_rest_days']
       df['total_rest'] = df['home_rest_days'] + df['away_rest_days']
       
       # Flag short/long rest
       df['home_short_rest'] = (df['home_rest_days'] < 6).astype(int)
       df['away_short_rest'] = (df['away_rest_days'] < 6).astype(int)
       df['home_long_rest'] = (df['home_rest_days'] > 8).astype(int)
       df['away_long_rest'] = (df['away_rest_days'] > 8).astype(int)
   
   # Game importance
   if 'excitement' in df.columns:
       df['rivalry_game'] = df.groupby(['homeTeamId', 'awayTeamId'])['excitement'].transform(
           lambda x: x > x.mean() + x.std()
       ).astype(int)
       
       df['big_game'] = df['excitement'] > df['excitement'].quantile(0.75)
       
   return df

def add_enhanced_momentum_features(games_df):
   """Add comprehensive momentum features"""
   df = games_df.copy()
   df = df.sort_values(['startDate', 'homeTeamId'])  
   
   for team_type in ['home', 'away']:
       points_col = f'{team_type}Points'
       team_id_col = f'{team_type}TeamId'
       opp_type = 'away' if team_type == 'home' else 'home'
       
       df[points_col] = pd.to_numeric(df[points_col], errors='coerce')
       
       # Short/long-term scoring
       for window in [3, 6]:
           df[f'{team_type}_points_last{window}'] = df.groupby([team_id_col, 'season'])[points_col].transform(
               lambda x: x.astype(float).rolling(window, min_periods=1).mean()
           )
           
       # Point trends
       df[f'{team_type}_points_momentum'] = (
           df[f'{team_type}_points_last3'] - df[f'{team_type}_points_last6']
       )
       
       # Win/Loss records
       df[f'{team_type}_won'] = (
           df[points_col] > df[f'{opp_type}Points']
       ).astype(int)
       
       # Point differential and scoring efficiency 
       df[f'{team_type}_point_diff'] = df[points_col] - df[f'{opp_type}Points']
       df[f'{team_type}_scoring_efficiency'] = df[points_col] / df.groupby([team_id_col, 'season'])[points_col].transform('mean')
       
       # Weighted recent performance
       df[f'{team_type}_win_quality'] = df.groupby([team_id_col, 'season']).apply(
           lambda x: (
               x[f'{team_type}_point_diff'] * 
               x[f'{team_type}_won'] * 
               np.exp(-np.arange(len(x))[::-1]/3)  # Exponential decay
           ).rolling(3, min_periods=1).mean()
       ).reset_index(level=[0,1], drop=True)
       
       # Form indicators
       df[f'{team_type}_in_form'] = (
           df[f'{team_type}_points_momentum'] > 0
       ).astype(int)
       
   return df

def add_sos_features(games_df):
    """Add strength of schedule features"""
    df = games_df.copy()
    
    metrics_mapping = {
        'elo': {'home': 'elo', 'away': 'elo_away'},
        'fpi': {'home': 'fpi', 'away': 'fpi_away'},
        'srs': {'home': 'srs', 'away': 'srs_away'}
    }
    
    for team_type in ['home', 'away']:
        team_id_col = f'{team_type}TeamId'
        opp_type = 'away' if team_type == 'home' else 'home'
        conf_col = f'{team_type}_conference'
        
        # Only process metrics that exist in the data
        for metric, cols in metrics_mapping.items():
            team_col = cols[team_type]
            if team_col in df.columns:
                # Calculate opponent strength
                df[f'{team_type}_opponent_{metric}'] = df.groupby([team_id_col, 'season'])[team_col].transform(
                    lambda x: x.expanding().mean().fillna(x.mean())
                )
                
                # Conference strength (if conference column exists)
                if conf_col in df.columns:
                    df[f'{team_type}_conf_{metric}'] = df.groupby([conf_col, 'season'])[team_col].transform(
                        lambda x: x.mean()
                    )
                    
                    # Conference rank (only for elo)
                    if metric == 'elo':
                        df[f'{team_type}_conf_rank'] = df.groupby('season')[f'{team_type}_conf_{metric}'].transform(
                            lambda x: x.rank(ascending=False)
                        )
        
        # Additional strength metrics if available
        epa_cols = [col for col in df.columns if 'epa' in col.lower() and team_type in col.lower()]
        for epa_col in epa_cols:
            if 'rolling' in epa_col:  # Only use rolling averages
                metric_name = epa_col.replace(f'{team_type}_', '').replace('_rolling_5', '')
                df[f'{team_type}_opponent_{metric_name}'] = df.groupby([team_id_col, 'season'])[epa_col].transform(
                    lambda x: x.expanding().mean().fillna(x.mean())
                )
    
    return df

def add_interaction_features(games_df):
    """Add interaction terms between features"""
    df = games_df.copy()
    
    # Metric differences
    metric_pairs = [
        ('elo', 'elo_away'),
        ('fpi', 'fpi_away'),
        ('srs', 'srs_away'),
        ('home_epa_rolling_5', 'away_epa_rolling_5'),
        ('home_success_rolling_5', 'away_success_rolling_5'),
        ('home_explosiveness_rolling_5', 'away_explosiveness_rolling_5')
    ]
    
    for home_col, away_col in metric_pairs:
        if home_col in df.columns and away_col in df.columns:
            diff_name = f"{home_col.replace('home_', '').replace('_away', '')}_diff"
            df[diff_name] = df[home_col] - df[away_col]
    
    # Momentum interactions
    epa_cols = [col for col in df.columns if 'epa' in col.lower() and 'rolling' in col.lower()]
    for col in epa_cols:
        if 'home_' in col and col.replace('home_', 'away_') in df.columns:
            momentum_name = f"momentum_{col.replace('home_', '').replace('_rolling_5', '')}"
            df[momentum_name] = df[col] - df[col.replace('home_', 'away_')]
    
    # Rest impact if available
    if 'home_rest_days' in df.columns and 'away_rest_days' in df.columns:
        df['rest_advantage'] = df['home_rest_days'] - df['away_rest_days']
        
        # Interact rest advantage with performance metrics
        for metric in ['elo_diff', 'fpi_diff', 'srs_diff']:
            if metric in df.columns:
                df[f'rest_{metric}'] = df['rest_advantage'] * df[metric]
    
    # Conference strength interactions
    if 'home_conference' in df.columns and 'away_conference' in df.columns:
        conf_metrics = [col for col in df.columns if 'conf_' in col and 'home_' in col]
        for col in conf_metrics:
            away_col = col.replace('home_', 'away_')
            if away_col in df.columns:
                df[f"{col.replace('home_', '')}_diff"] = df[col] - df[away_col]
    
    return df

def create_conference_models(X_train, y_train, is_home=True):
    """Create separate models for each conference"""
    conf_models = {}
    conf_col = 'home_conference_encoded' if is_home else 'away_conference_encoded'
    
    for conf_id in X_train[conf_col].unique():
        mask = X_train[conf_col] == conf_id
        if mask.sum() > 100:
            X_conf = X_train[mask]
            y_conf = y_train[mask]
            model = RandomForestRegressor(**BASE_RF_CONFIG)
            try:
                model.fit(X_conf, y_conf)
                conf_models[conf_id] = {
                    'model': model,
                    'samples': len(X_conf),
                    'features': X_conf.shape[1]
                }
            except Exception as e:
                print(f"Error training conference model {conf_id}: {str(e)}")
    return conf_models

def create_seasonal_models(X_train, y_train, week_col='week'):
    """Create separate models for different parts of season"""
    bins = pd.qcut(X_train[week_col].astype(float), q=3, labels=False)
    period_models = {}
    
    for period_idx, period_name in enumerate(['Early', 'Mid', 'Late']):
        mask = bins == period_idx
        if mask.sum() > 100:
            X_period = X_train[mask]
            y_period = y_train[mask]
            model = RandomForestRegressor(**BASE_RF_CONFIG)
            try:
                model.fit(X_period, y_period)
                period_models[period_name] = {
                    'model': model,
                    'samples': len(X_period),
                    'features': X_period.shape[1]
                }
            except Exception as e:
                print(f"Error training {period_name} season model: {str(e)}")
    return period_models

def create_yearly_models(X_train, y_train, is_home=True):
    """Create separate models for each year"""
    yearly_models = {}
    for year in sorted(X_train['season'].unique()):
        mask = X_train['season'] == year
        if mask.sum() > 100:
            X_year = X_train[mask]
            y_year = y_train[mask]
            model = RandomForestRegressor(**BASE_RF_CONFIG)
            imputer = SimpleImputer(strategy='median')
            X_year_imputed = imputer.fit_transform(X_year)
            try:
                model.fit(X_year_imputed, y_year)
                yearly_models[year] = {
                    'model': model,
                    'imputer': imputer,
                    'samples': len(X_year),
                    'features': X_year.shape[1]
                }
            except Exception as e:
                print(f"Error training {year} model: {str(e)}")
    return yearly_models

def evaluate_predictions(y_true, y_pred, prefix=''):
    """Calculate evaluation metrics"""
    metrics = {
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred)
    }
    diff = np.abs(y_true - y_pred)
    for margin in [3, 7, 10, 14]:
        metrics[f'within_{margin}'] = (diff <= margin).mean()
    return metrics
    
def evaluate_yearly_models(yearly_models, X_test, y_test):
    """Evaluate performance of yearly models"""
    yearly_metrics = {}
    
    for year, model_info in yearly_models.items():
        if isinstance(model_info, dict) and 'model' in model_info and 'imputer' in model_info:
            # Transform test data using the year's imputer
            X_test_imputed = model_info['imputer'].transform(X_test)
            
            # Get predictions
            y_pred = model_info['model'].predict(X_test_imputed)
            
            # Calculate metrics
            yearly_metrics[year] = {
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'mae': mean_absolute_error(y_test, y_pred),
                'r2': r2_score(y_test, y_pred)
            }
            
    return yearly_metrics
def train_evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    """Train and evaluate a single model"""
    imputer = SimpleImputer(strategy='median')
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)
    
    model.fit(X_train_imputed, y_train)
    return {
        'model': model,
        'imputer': imputer,
        'train_metrics': evaluate_predictions(y_train, model.predict(X_train_imputed)),
        'test_metrics': evaluate_predictions(y_test, model.predict(X_test_imputed)),
        'train_predictions': model.predict(X_train_imputed),
        'test_predictions': model.predict(X_test_imputed)
    }

def create_stratified_season_split(games_df, test_size=0.2, random_state=42):
    """Create train/test splits preserving season structure"""
    train_mask = pd.Series(False, index=games_df.index)
    test_mask = pd.Series(False, index=games_df.index)
    
    for season in games_df['season'].unique():
        season_mask = games_df['season'] == season
        season_weeks = games_df[season_mask]['week'].unique()
        n_test_weeks = max(1, int(len(season_weeks) * test_size))
        
        np.random.seed(random_state + season)
        test_weeks = np.random.choice(season_weeks, size=n_test_weeks, replace=False)
        
        train_mask |= season_mask & ~games_df['week'].isin(test_weeks)
        test_mask |= season_mask & games_df['week'].isin(test_weeks)
    
    return train_mask, test_mask

def prepare_features(games_df, encoders=None):
    """Prepare features for model input"""
    try:
        df = games_df.copy()
        
        # Features to exclude
        exclude_cols = [
            'homePoints', 'awayPoints', 'homeTeam', 'awayTeam', 'homeTeamId',
            'awayTeamId', 'startDate', 'id', 'gameId', 'awayLineScores',
            'homeLineScores', 'homeEndElo', 'awayEndElo', 'homePostgameWinProb',
            'awayPostgameWinProb', 'away_won', 'home_won'
        ]
        
        # Handle missing values
        for col in df.columns:
            if col in exclude_cols:
                continue
            elif any(x in col.lower() for x in ['rolling', 'elo', 'fpi', 'sp', 'srs']):
                df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
            elif 'days_rest' in col.lower():
                df[col] = df[col].fillna(7)  
            elif any(x in col.lower() for x in ['streak', 'h2h', 'win', 'momentum', 'quality', 'sos']):
                df[col] = df[col].fillna(0)
            elif any(x in col.lower() for x in ['historical', 'venue']):
                df[col] = df[col].fillna(df[col].mean())  # Fixed: was using data[col].mean()
        
        # Prepare feature sets
        features_df = df[[col for col in df.columns if col not in exclude_cols]].copy()
        numeric_cols = features_df.select_dtypes(include=['int64', 'float64']).columns
        categorical_cols = features_df.select_dtypes(include=['object']).columns
        
        encoded_data = features_df[numeric_cols].copy()
        current_encoders = encoders or {}
        
        # Encode categorical features
        for col in categorical_cols:
            col_data = features_df[col].fillna('unknown').astype(str)
            
            if encoders is None:
                encoder = LabelEncoder()
                encoder.fit(np.append(col_data.unique(), 'unknown'))
                current_encoders[col] = encoder
            else:
                encoder = encoders[col]
                col_data = pd.Series(
                    np.where(~col_data.isin(encoder.classes_), 'unknown', col_data),
                    index=col_data.index
                )
                
            encoded_data[f'{col}_encoded'] = encoder.transform(col_data)
            
        encoded_data = encoded_data.fillna(encoded_data.median())
        
        return (encoded_data, current_encoders) if encoders is None else encoded_data
        
    except Exception as e:
        print(f"Error preparing features: {str(e)}")
        raise

def add_enhanced_features(games_df):
    """Master feature engineering function"""
    if games_df.empty:
        return games_df.copy()
        
    enhanced_df = games_df.copy()
    enhanced_df['startDate'] = pd.to_datetime(enhanced_df['startDate'])

    print("Adding team performance context...")
    enhanced_df = add_team_performance_context(enhanced_df)
    
    print("Adding timing features...")  
    enhanced_df = add_enhanced_timing_features(enhanced_df)
    
    print("Adding momentum features...")
    enhanced_df = add_enhanced_momentum_features(enhanced_df)
    
    print("Adding strength of schedule features...")
    enhanced_df = add_sos_features(enhanced_df)
    
    print("Adding feature interactions...")
    enhanced_df = add_interaction_features(enhanced_df)
    
    # Add season context
    if len(enhanced_df) > 0 and 'week' in enhanced_df.columns:
        enhanced_df['season_period'] = pd.qcut(enhanced_df['week'], q=3, labels=['Early', 'Mid', 'Late'])
    else:
        enhanced_df['season_period'] = 'Mid'
        
    # Add expected scoring if available
    if 'overUnder' in enhanced_df.columns:
        enhanced_df['expected_scoring'] = enhanced_df['overUnder'].fillna(enhanced_df['overUnder'].mean())
        
        if 'fpiOffensiveEfficiency' in enhanced_df.columns:
            enhanced_df['scoring_efficiency_interaction'] = (
                enhanced_df['expected_scoring'] * 
                enhanced_df['fpiOffensiveEfficiency'] * 
                np.where(enhanced_df['homeTeam'].notna(), 1.1, 1.0)
            )
            
    return enhanced_df

enhanced_df = add_enhanced_features(games_df)


# %%
# Check if data is loaded
print("Games DataFrame loaded:", games_df is not None)
print("Games DataFrame shape:", games_df.shape if games_df is not None else "Not loaded")

# Check if data is loaded
print("Games DataFrame loaded:", enhanced_df is not None)
print("Games DataFrame shape:", enhanced_df.shape if enhanced_df is not None else "Not loaded")

# %% [markdown]
# # Model Training

# %%
RANDOM_STATE = 42
BASE_RF_CONFIG = {
    'n_estimators': 500,
    'max_depth': 15,
    'min_samples_split': 10,
    'min_samples_leaf': 5,
    'random_state': RANDOM_STATE,
    'n_jobs': -1
}

def train_models(enhanced_df, TRAIN_SIZE=0.8, RANDOM_STATE=42):
    base_models = {
        'rf': RandomForestRegressor(**BASE_RF_CONFIG),
        'gbm': GradientBoostingRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            random_state=RANDOM_STATE
        ),
        'xgb': XGBRegressor(
            n_estimators=1, 
            max_depth=8,
            learning_rate=0.01,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
    }

    # Prepare data splits
    home_data = prepare_team_data(enhanced_df, 'home', TRAIN_SIZE)
    X_home_train, X_home_test, y_home_train, y_home_test, home_encoders = home_data
    
    away_data = prepare_team_data(enhanced_df, 'away', TRAIN_SIZE, home_encoders)
    X_away_train, X_away_test, y_away_train, y_away_test, _ = away_data
    
    # Train models
    results = {'home': {}, 'away': {}}
    
    for model_type, base_model in base_models.items():
        # Train home model
        home_model = train_evaluate_model(
            base_model, X_home_train, X_home_test, 
            y_home_train, y_home_test, f"Home-{model_type}"
        )
        results['home'][model_type] = home_model

        # Train away model  
        away_model = train_evaluate_model(
            clone(base_model), X_away_train, X_away_test,
            y_away_train, y_away_test, f"Away-{model_type}" 
        )
        results['away'][model_type] = away_model

    # Train specialized models with clean data
    conf_models = {
        'home': create_conference_models(X_home_train, y_home_train, True),
        'away': create_conference_models(X_away_train, y_away_train, False)
    }
    
    season_models = {
        'home': create_seasonal_models(X_home_train, y_home_train),
        'away': create_seasonal_models(X_away_train, y_away_train)
    }
    
    yearly_models = {
        'home': create_yearly_models(X_home_train, y_home_train, True),
        'away': create_yearly_models(X_away_train, y_away_train, False)
    }
    
    # Evaluate yearly models
    yearly_metrics = {
        'home': evaluate_yearly_models(yearly_models['home'], X_home_test, y_home_test),
        'away': evaluate_yearly_models(yearly_models['away'], X_away_test, y_away_test)
    }

    # Compile model info
    model_info = {
        'models': {
            'home': {'rf': results['home']['rf']},
            'away': {'rf': results['away']['rf']}
        },
        'performances': summarize_performances(results),
        'encoders': home_encoders,
        'features': X_home_train.columns.tolist(),
        'specialized_models': {
            'conference': conf_models,
            'seasonal': season_models,
            'yearly': yearly_models,
            'yearly_metrics': yearly_metrics
        }
    }
    
    print_performance_summary(model_info)
    return model_info

def prepare_team_data(enhanced_df, team_type, train_size, encoders=None):
    """Prepare data for modeling"""
    team_id_col = f'{team_type}TeamId'
    points_col = f'{team_type}Points'
    
    # Filter out invalid rows
    valid_idx = (enhanced_df[points_col].notna()) & (enhanced_df[team_id_col].notna())
    X = enhanced_df[valid_idx].copy()
    y = X[points_col].copy()
    
    # Split train/test
    train_mask, test_mask = create_stratified_season_split(X, 1-train_size)
    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    
    # Prepare features
    if encoders is None:
        X_train_prep, encoders = prepare_features(X_train)
        X_test_prep = prepare_features(X_test, encoders)
    else:
        X_train_prep = prepare_features(X_train, encoders) 
        X_test_prep = prepare_features(X_test, encoders)
        
    return X_train_prep, X_test_prep, y_train, y_test, encoders

def summarize_performances(results):
   """Summarize model performances"""
   performances = {'home': {}, 'away': {}}
   
   for team_type in ['home', 'away']:
       for name, res in results[team_type].items():
           if isinstance(res, dict) and 'test_metrics' in res:
               metrics = res['test_metrics']
               performances[team_type][name] = {
                   'rmse': metrics['rmse'],
                   'mae': metrics['mae'], 
                   'r2': metrics['r2'],
                   'within_7': metrics['within_7'],
                   'within_10': metrics['within_10']
               }
   
   return performances

def print_performance_summary(model_info):
   """Print model performance summary"""
   for point_type in ['home', 'away']:
       print(f"\n{point_type.capitalize()} Models Performance:")
       for model_name, metrics in model_info['performances'][point_type].items():
           print(f"\n{model_name.upper()}:")
           print(f"RMSE: {metrics['rmse']:.2f}")
           print(f"MAE: {metrics['mae']:.2f}") 
           print(f"R² Score: {metrics['r2']:.2f}")
           print(f"Within 7: {metrics['within_7']:.1%}")
           print(f"Within 10: {metrics['within_10']:.1%}")
           
   for point_type in ['home', 'away']:
       if model_info['specialized_models']['yearly_metrics'][point_type]:
           print(f"\n{point_type.title()} Yearly Performance:")
           metrics = model_info['specialized_models']['yearly_metrics'][point_type]
           for year, year_metrics in metrics.items():
               print(f"\nYear {year}:")
               print(f"RMSE: {year_metrics['rmse']:.2f}")
               print(f"MAE: {year_metrics['mae']:.2f}")
               print(f"R²: {year_metrics['r2']:.2f}")

def save_models(results, best_home_model_name, best_away_model_name, home_encoders, enhanced_df):
    """Save trained models and data"""
    try:
        current_dir = os.getcwd()
        save_path = os.path.join(current_dir, 'saved')
        data_path = os.path.join(current_dir, 'data')
        
        os.makedirs(save_path, exist_ok=True)
        os.makedirs(data_path, exist_ok=True)
        
        # Get results using correct structure (accessing 'models' first)
        home_results = results['models']['home']['rf']  # Using rf as that's what's in your structure
        away_results = results['models']['away']['rf']
        
        try:
            print("Saving home model...")
            joblib.dump(home_results['model'], os.path.join(save_path, 'home_points_model.joblib'))
        except Exception as e:
            print(f"Error saving home model: {e}")
            raise
        
        try:
            print("Saving home metrics...")
            joblib.dump(home_results['test_metrics'], os.path.join(save_path, 'home_points_metrics.joblib'))
        except Exception as e:
            print(f"Error saving home metrics: {e}")
            raise

        try:
            print("Saving away model...")
            joblib.dump(away_results['model'], os.path.join(save_path, 'away_points_model.joblib'))
        except Exception as e:
            print(f"Error saving away model: {e}")
            raise

        try:
            print("Saving away metrics...")
            joblib.dump(away_results['test_metrics'], os.path.join(save_path, 'away_points_metrics.joblib'))
        except Exception as e:
            print(f"Error saving away metrics: {e}")
            raise

        try:
            print("Saving imputer...")
            joblib.dump(home_results['imputer'], os.path.join(save_path, 'imputer.joblib'))
        except Exception as e:
            print(f"Error saving imputer: {e}")
            raise

        try:
            print("Saving encoders...")
            joblib.dump(results['encoders'], os.path.join(save_path, 'encoders.joblib'))
        except Exception as e:
            print(f"Error saving encoders: {e}")
            raise

        try:
            print("Saving features...")
            joblib.dump(results['features'], os.path.join(save_path, 'features.joblib'))
        except Exception as e:
            print(f"Error saving features: {e}")
            raise

        try:
            print("Saving processed data...")
            enhanced_df.to_csv(os.path.join(data_path, 'processed_games.csv'), index=False)
        except Exception as e:
            print(f"Error saving processed data: {e}")
            raise

        print("\nAll models and data saved successfully!")
        return True

    except Exception as e:
        print(f"\nFatal error during save process: {e}")
        return False
   

# Execute training
model_info = train_models(enhanced_df)

save_models(
    results=model_info,
    best_home_model_name='gbm',
    best_away_model_name='gbm',
    home_encoders=model_info['encoders'],
    enhanced_df=enhanced_df
)

# %% [markdown]
# # Model Evaluation

# %%
# Model Evaluation Functions

def validate_predictions(results, y_home_test, y_away_test):
   """Validate model predictions vs actuals"""
   home_preds = results['home']['rf']['test_predictions'] 
   away_preds = results['away']['rf']['test_predictions']
   
   print("\nPrediction Stats:")
   print(f"Home - Predicted avg: {home_preds.mean():.1f} (Actual: {y_home_test.mean():.1f})")
   print(f"Away - Predicted avg: {away_preds.mean():.1f} (Actual: {y_away_test.mean():.1f})")
   print(f"Home advantage: {(home_preds - away_preds).mean():.1f} (Actual: {(y_home_test - y_away_test).mean():.1f})")
   
   print("\nRanges:")
   print(f"Home predictions: {home_preds.min():.1f} to {home_preds.max():.1f}")
   print(f"Home actuals: {y_home_test.min():.1f} to {y_home_test.max():.1f}")
   print(f"Away predictions: {away_preds.min():.1f} to {away_preds.max():.1f}") 
   print(f"Away actuals: {y_away_test.min():.1f} to {y_away_test.max():.1f}")
   
   unrealistic = {
       'home': sum((home_preds < 0) | (home_preds > 100)),
       'away': sum((away_preds < 0) | (away_preds > 100))
   }
   print(f"\nUnrealistic scores (outside 0-100):")
   print(f"Home: {unrealistic['home']}")
   print(f"Away: {unrealistic['away']}")

def analyze_features_comprehensive(model, feature_names, top_n=20):
   """Analyze feature importance and feature groups"""
   if not hasattr(model, 'feature_importances_'):
       return None, None
       
   importances = pd.DataFrame({
       'feature': feature_names,
       'importance': model.feature_importances_
   }).sort_values('importance', ascending=False)
   
   feature_groups = {
       'Time-weighted': ['last3', 'season_weighted', 'trend', 'rolling', 'momentum'],
       'Team-specific': ['win_pct', 'conf_dominance', 'field_impact', 'road_perf', 'venue'],  
       'Base Stats': ['points', 'elo', 'fpi', 'epa', 'success'],
       'Context': ['week', 'season', 'rest', 'weather'],
       'Identity': ['team', 'school', 'conf', 'division']
   }
   
   group_importance = {}
   print("\nFeature Groups:")
   for group, keywords in feature_groups.items():
       mask = importances['feature'].str.contains('|'.join(keywords), case=False)
       group_importance[group] = importances[mask]['importance'].sum()
       print(f"{group}: {group_importance[group]:.4f}")
       
   plot_feature_importance_groups(importances, group_importance, top_n)
   return importances, group_importance
   
def plot_feature_importance_groups(importances, group_importance, top_n):
   """Plot feature importance analysis"""
   plt.figure(figsize=(15, 10))
   
   plt.subplot(2, 1, 1)
   plt.title('Top Features')
   sns.barplot(data=importances.head(top_n), x='importance', y='feature')
   
   plt.subplot(2, 1, 2) 
   plt.title('Feature Groups')
   group_df = pd.DataFrame({
       'group': list(group_importance.keys()),
       'importance': list(group_importance.values())
   }).sort_values('importance', ascending=False)
   sns.barplot(data=group_df, x='importance', y='group')
   
   plt.tight_layout()
   plt.show()

def plot_feature_importance(model, features, title, top_n=20):
   """Plot feature importance ranking"""
   if not hasattr(model, 'feature_importances_'):
       return None
       
   importance = pd.DataFrame({
       'feature': features,
       'importance': model.feature_importances_
   }).sort_values('importance', ascending=False)
   
   plt.figure(figsize=(12, 6))
   plt.title(title)
   sns.barplot(data=importance.head(top_n), x='importance', y='feature')
   plt.xlabel('Importance')
   plt.tight_layout()
   plt.show()
   
   return importance

def evaluate_spread_predictions(y_home, y_away, y_home_pred, y_away_pred): 
   """Evaluate spread and total predictions"""
   actual_spread = y_home - y_away
   pred_spread = y_home_pred - y_away_pred
   actual_total = y_home + y_away
   pred_total = y_home_pred + y_away_pred
   
   spread_metrics = {
       'rmse': np.sqrt(mean_squared_error(actual_spread, pred_spread)),
       'mae': mean_absolute_error(actual_spread, pred_spread),
       'directional_accuracy': (np.sign(actual_spread) == np.sign(pred_spread)).mean()
   }
   
   total_metrics = {
       'rmse': np.sqrt(mean_squared_error(actual_total, pred_total)),
       'mae': mean_absolute_error(actual_total, pred_total)
   }
   
   # Scoring brackets
   for low, high in [(0,35), (36,50), (51,65), (66,80), (81,100)]:
       mask = (actual_total >= low) & (actual_total <= high)
       if mask.sum() > 0:
           total_metrics[f'rmse_{low}_{high}'] = np.sqrt(mean_squared_error(
               actual_total[mask], pred_total[mask]
           ))
   
   print("\nSpread Metrics:") 
   for metric, value in spread_metrics.items():
       if 'accuracy' in metric:
           print(f"{metric.replace('_', ' ').title()}: {value:.1%}")
       else:
           print(f"{metric.upper()}: {value:.2f}")
   
   print("\nTotal Points Metrics:")
   print(f"RMSE: {total_metrics['rmse']:.2f}") 
   print(f"MAE: {total_metrics['mae']:.2f}")
   print("\nRMSE by Scoring Bracket:")
   for low, high in [(0,35), (36,50), (51,65), (66,80), (81,100)]:
       key = f'rmse_{low}_{high}'
       if key in total_metrics:
           print(f"{low}-{high} points: {total_metrics[key]:.2f}")
           
   return spread_metrics, total_metrics

def compare_model_feature_importance(home_importance, away_importance, home_groups, away_groups):
   """Compare home vs away model feature importance"""
   print("\nFeature Importance Comparison:")
   
   all_groups = sorted(set(home_groups) | set(away_groups))
   comparison_data = [{
       'group': group,
       'home_importance': home_groups.get(group, 0),
       'away_importance': away_groups.get(group, 0),
       'difference': home_groups.get(group, 0) - away_groups.get(group, 0)
   } for group in all_groups]
   
   comparison_df = pd.DataFrame(comparison_data).sort_values('difference', ascending=False)
   print("\nGroup Importance Differences:")
   print(comparison_df)
   
   plot_importance_comparisons(comparison_df, all_groups, home_importance, away_importance)
   
   compare_top_features(home_importance, away_importance)

def plot_importance_comparisons(comparison_df, all_groups, home_importance, away_importance):
   """Plot feature importance comparisons"""
   plt.figure(figsize=(15, 8))
   x = np.arange(len(all_groups))
   width = 0.35
   
   plt.bar(x - width/2, comparison_df['home_importance'], width, label='Home')
   plt.bar(x + width/2, comparison_df['away_importance'], width, label='Away')
   
   plt.xlabel('Feature Groups')
   plt.ylabel('Importance')
   plt.title('Group Importance: Home vs Away')
   plt.xticks(x, all_groups, rotation=45)
   plt.legend()
   plt.tight_layout()
   plt.show()
   
   plot_top_feature_comparison(home_importance, away_importance)

def compare_top_features(home_importance, away_importance):
   """Compare top features between models"""
   print("\nTop Feature Comparison:")
   top_features = pd.merge(
       home_importance.head(10).rename(columns={'importance': 'home_imp'}),
       away_importance.head(10).rename(columns={'importance': 'away_imp'}),
       on='feature',
       how='outer'
   ).fillna(0)
   
   top_features['diff'] = top_features['home_imp'] - top_features['away_imp']
   print("\nFeature Importance Differences:")
   print(top_features.sort_values('diff', ascending=False))
   
def plot_top_feature_comparison(home_importance, away_importance):
   """Plot top feature comparison"""
   plt.figure(figsize=(15, 8))
   
   features = list(set(home_importance.head(10)['feature']) | 
                  set(away_importance.head(10)['feature']))
                  
   home_vals = [home_importance[home_importance['feature'] == f]['importance'].iloc[0] 
                if f in home_importance['feature'].values else 0 for f in features]
   away_vals = [away_importance[away_importance['feature'] == f]['importance'].iloc[0]
                if f in away_importance['feature'].values else 0 for f in features]
   
   x = np.arange(len(features))
   width = 0.35
   plt.bar(x - width/2, home_vals, width, label='Home')
   plt.bar(x + width/2, away_vals, width, label='Away')
   
   plt.xlabel('Features')
   plt.ylabel('Importance')
   plt.title('Top Feature Importance: Home vs Away')
   plt.xticks(x, features, rotation=45, ha='right')
   plt.legend()
   plt.tight_layout()
   plt.show()

# Main evaluation code
def evaluate_models(results, model_info, X_home_train, y_home_test, y_away_test):
   """Run full model evaluation"""
   validate_predictions(results, y_home_test, y_away_test)
   
   print("\nAnalyzing Feature Importance...")
   if 'rf' in results['home'] and 'rf' in results['away']:
       home_importance, home_groups = analyze_features_comprehensive(
           results['home']['rf']['model'],
           X_home_train.columns
       )
       away_importance, away_groups = analyze_features_comprehensive(  
           results['away']['rf']['model'],
           X_home_train.columns
       )
       compare_model_feature_importance(
           home_importance,
           away_importance, 
           home_groups,
           away_groups
       )
   
   print("\nModel Performance Summary:")
   # Find best models based on RMSE
   best_home_model = None
   best_away_model = None
   best_home_rmse = float('inf')
   best_away_rmse = float('inf')
   
   for point_type in ['home', 'away']:
       print(f"\n{point_type.title()} Models:")
       for model_name, metrics in model_info['performances'][point_type].items():
           print(f"\n{model_name.upper()}:")
           print(f"RMSE: {metrics['rmse']:.2f}")
           print(f"MAE: {metrics['mae']:.2f}")
           print(f"R²: {metrics['r2']:.2f}")
           print(f"Within 7: {metrics['within_7']:.1%}")
           print(f"Within 10: {metrics['within_10']:.1%}")
           
           # Track best models
           if point_type == 'home' and metrics['rmse'] < best_home_rmse:
               best_home_rmse = metrics['rmse']
               best_home_model = model_name
           elif point_type == 'away' and metrics['rmse'] < best_away_rmse:
               best_away_rmse = metrics['rmse']
               best_away_model = model_name

   print(f"\nBest Home Model: {best_home_model} (RMSE: {best_home_rmse:.2f})")
   print(f"Best Away Model: {best_away_model} (RMSE: {best_away_rmse:.2f})")

   # Evaluate spread predictions using best models
   if best_home_model and best_away_model and \
      best_home_model in results['home'] and best_away_model in results['away']:
       print("\nSpread Predictions using Best Models:")
       evaluate_spread_predictions(
           y_home_test,
           y_away_test,
           results['home'][best_home_model]['test_predictions'],
           results['away'][best_away_model]['test_predictions']
       )

# %% [markdown]
# # Prediction Interface

# %%
def calculate_dynamic_weights(home_team, away_team, home_recent, away_recent, enhanced_df):
   """
   Calculate dynamic weights based on team characteristics and data quality
   """
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

# %%
home_team = 'Wyoming'
away_away = 'Ohio State'

def display_prediction(home_team, away_team, home_score, away_score, details):
    """Display prediction results in a clear, formatted way"""
    # Calculate spread and determine favorite
    spread = home_score - away_score
    total = home_score + away_score
    
    # Determine favorite and spread display
    if spread < 0:
        favorite = away_team
        spread_text = f"{away_team} {+spread:+.1f}"
    else:
        favorite = home_team
        spread_text = f"{home_team} {-spread:+.1f}"
    
    print("\n🏈 Game Prediction")
    print("─" * 50)
    print(f"{home_team:20} {home_score:.1f}")
    print(f"{away_team:20} {away_score:.1f}")
    print("─" * 50)
    print(f"{'Spread:':<20} {spread_text}")
    print(f"{'Total:':<20} {total:.1f}")
    
    print("\n📊 Key Factors")
    print("─" * 50)
    
    # [Rest of the function remains the same]
    
    # Base Performance
    print("\nBase Performance:")
    print(f"{'Home Scoring Avg:':<25} {details['base_scores']['home']:.1f}")
    print(f"{'Away Scoring Avg:':<25} {details['base_scores']['away']:.1f}")
    
    # Team Strengths
    print("\nTeam Strength Analysis:")
    print(f"{'Home Team Strength:':<25} {details['strength']['home']:+.1f}")
    print(f"{'Away Team Strength:':<25} {details['strength']['away']:+.1f}")
    
    # Form
    print("\nRecent Form:")
    print(f"{'Home Team Form:':<25} {details['form']['home']:+.1f}")
    print(f"{'Away Team Form:':<25} {details['form']['away']:+.1f}")
    
    # Conference Analysis
    print("\nConference Analysis:")
    print(f"{'Home Conference Avg:':<25} {details['conf_stats']['home']['points']:.1f}")
    print(f"{'Away Conference Avg:':<25} {details['conf_stats']['away']['points']:.1f}")
    
    # Impact Factors
    print("\nImpact Factors:")
    print(f"{'Rankings Impact:':<25} {details['rankings']['impact']:+.1f}")
    print(f"{'Efficiency Impact:':<25} {details['efficiency']['diff']:+.1f}")
    print(f"{'H2H Factor:':<25} {details['h2h_factor']:+.1f}")
    print(f"{'Weather Impact:':<25} {details['weather']['impact']:+.1f}")
    print(f"{'Rest Advantage:':<25} {details['rest']['advantage']:+.1f}")
    
    # Model Weights
    print("\n⚖️ Model Weights")
    print("─" * 50)
    for factor, weight in details['weights'].items():
        print(f"{factor.replace('_', ' ').title():<20} {weight:.2%}")

# Make prediction
home_score, away_score, details = predict_game_score(
    home_team='Wyoming', 
    away_team='Ohio State', 
    enhanced_df=enhanced_df  # Add the required DataFrame
)

# Display results
display_prediction('Wyoming', 'Ohio State', home_score, away_score, details)

# %% [markdown]
# # Save Models

# %%
def save_models(results, best_home_model_name, best_away_model_name, home_encoders, enhanced_df):
    """Save trained models and data"""
    try:
        current_dir = os.getcwd()
        save_path = os.path.join(current_dir, 'saved')
        data_path = os.path.join(current_dir, 'data')
        
        os.makedirs(save_path, exist_ok=True)
        os.makedirs(data_path, exist_ok=True)
        
        # Get results using correct structure (accessing 'models' first)
        home_results = results['models']['home']['rf']  # Using rf as that's what's in your structure
        away_results = results['models']['away']['rf']
        
        try:
            print("Saving home model...")
            joblib.dump(home_results['model'], os.path.join(save_path, 'home_points_model.joblib'))
        except Exception as e:
            print(f"Error saving home model: {e}")
            raise
        
        try:
            print("Saving home metrics...")
            joblib.dump(home_results['test_metrics'], os.path.join(save_path, 'home_points_metrics.joblib'))
        except Exception as e:
            print(f"Error saving home metrics: {e}")
            raise

        try:
            print("Saving away model...")
            joblib.dump(away_results['model'], os.path.join(save_path, 'away_points_model.joblib'))
        except Exception as e:
            print(f"Error saving away model: {e}")
            raise

        try:
            print("Saving away metrics...")
            joblib.dump(away_results['test_metrics'], os.path.join(save_path, 'away_points_metrics.joblib'))
        except Exception as e:
            print(f"Error saving away metrics: {e}")
            raise

        try:
            print("Saving imputer...")
            joblib.dump(home_results['imputer'], os.path.join(save_path, 'imputer.joblib'))
        except Exception as e:
            print(f"Error saving imputer: {e}")
            raise

        try:
            print("Saving encoders...")
            joblib.dump(results['encoders'], os.path.join(save_path, 'encoders.joblib'))
        except Exception as e:
            print(f"Error saving encoders: {e}")
            raise

        try:
            print("Saving features...")
            joblib.dump(results['features'], os.path.join(save_path, 'features.joblib'))
        except Exception as e:
            print(f"Error saving features: {e}")
            raise

        try:
            print("Saving processed data...")
            enhanced_df.to_csv(os.path.join(data_path, 'processed_games.csv'), index=False)
        except Exception as e:
            print(f"Error saving processed data: {e}")
            raise

        print("\nAll models and data saved successfully!")
        return True

    except Exception as e:
        print(f"\nFatal error during save process: {e}")
        return False
   

def initialize_model():
    """Initialize the model and load necessary data"""
    global enhanced_df, model_info, home_encoders
    try:
        # Get the current working directory instead of using __file__
        import os
        current_dir = os.getcwd()
        save_path = os.path.join(current_dir, 'saved')
        data_path = os.path.join(current_dir, 'data')
        
        # Check if directories exist
        if not os.path.exists(save_path) or not os.path.exists(data_path):
            print(f"Required directories not found. Creating directories...")
            os.makedirs(save_path, exist_ok=True)
            os.makedirs(data_path, exist_ok=True)
            return False  # Return False if this is first run
        
        # Try to load the model files
        try:
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
            games_df = pd.read_csv(os.path.join(data_path, 'processed_games.csv'))
            games_df['startDate'] = pd.to_datetime(games_df['startDate'])
            
            # Set the enhanced_df
            enhanced_df = add_enhanced_features(games_df)
            
            # Load encoders
            home_encoders = model_info['encoders']
            
            return True
            
        except FileNotFoundError:
            print("Model files not found. Please run the training script first.")
            return False
            
    except Exception as e:
        print(f"Error initializing model: {str(e)}")
        print("Please ensure you've run the training script first to create the model files.")
        return False

# Initialize global variables
enhanced_df = None
model_info = None
home_encoders = None

# Initialize when module is imported
if initialize_model():
    print("Model initialized successfully")
else:
    print("Model initialization failed. Need to train the model first.")
    # Don't raise an error, just print a message since we're in a notebook

__all__ = [
    'enhanced_df', 
    'model_info', 
    'predict_game_score',  # This is the base function
    'predict_score',       # This is the UI-friendly version
    'prepare_features',
    'add_enhanced_features',
    'add_rolling_stats',
    'save_models'
]



