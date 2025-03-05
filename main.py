"""
Main Flask application file for portfolio website with various features including:
- Color generation
- CFB predictions
- Teaching assistant
- Reinforcement learning
- Revenue forecasting
"""

# Standard library imports - only import what's absolutely necessary
import os
import re
import json
import base64
import socket
import traceback
from typing import Dict, List
import time
import logging
from functools import lru_cache
import threading

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize modules dict and lock for thread-safe lazy loading
_modules = {}
_modules_lock = threading.Lock()

# Create the Flask app instance
from flask import Flask, request, render_template, jsonify, url_for, send_from_directory, abort
app = Flask(__name__, 
            static_folder='static', 
            static_url_path='/static',
            template_folder='templates')

# Configure app with minimal settings
app.config['SECRET_KEY'] = os.urandom(24)
app.config['PREFERRED_URL_SCHEME'] = 'https'
app.config['APPLICATION_ROOT'] = '/'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Cache control for static files
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 31536000  # 1 year in seconds

# Enable debug mode only in development
debug_mode = os.environ.get('FLASK_ENV') == 'development'
app.debug = debug_mode

# Enable secure headers
@app.after_request
def add_header(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'SAMEORIGIN'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    
    # Add caching headers for static files
    if request.endpoint == 'static':
        response.cache_control.max_age = 60 * 60 * 24 * 30  # 30 days
        response.cache_control.public = True
    
    # Ensure CORS headers are present for API calls
    if request.path.startswith('/api/'):
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = '*'
    
    return response

# Set up CORS - Light version
from flask_cors import CORS
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "allow_headers": "*",
        "expose_headers": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "supports_credentials": True
    }
})

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Create training manager class that doesn't rely on WebSockets
class TrainingManager:
    def __init__(self):
        self.is_training = False
        self.agent = None
        self.training_thread = None
        self.last_update = time.time()
        
    def start_training(self, agent):
        """Start training loop in a separate thread"""
        logger.info("Starting training loop")
        self.agent = agent
        agent.training_complete = False  # Reset training flag
        
        def training_loop():
            logger.info("Training loop started")
            
            while self.is_training:
                try:
                    # Check if training is complete
                    if agent.training_complete:
                        logger.info("Training complete! Stopping training loop.")
                        self.is_training = False
                        break

                    # Run training episode
                    agent.train_episode()
                    self.last_update = time.time()
                    
                    # Log progress every 10 episodes
                    metrics = agent.get_metrics()
                    if metrics['episode_count'] % 10 == 0:
                        logger.info(f"Episode {metrics['episode_count']}, success rate: {metrics['success_rate']:.1f}%")
                    
                    # Check if max episodes reached or success rate achieved
                    if metrics['training_complete']:
                        logger.info("Training complete by metrics! Stopping training loop.")
                        self.is_training = False
                        break
                    
                    # Control update frequency - don't update too fast
                    time.sleep(0.1)
                    
                except Exception as e:
                    logger.error(f"Error in training loop: {e}")
                    traceback.print_exc()
                    self.is_training = False
                    break
                
        self.is_training = True
        self.training_thread = threading.Thread(target=training_loop, daemon=True)
        self.training_thread.start()
        logger.info(f"Training thread started and is alive: {self.training_thread.is_alive()}")
        
    def stop_training(self):
        """Stop the training loop"""
        logger.info("Stopping training loop")
        self.is_training = False
        if self.training_thread and self.training_thread.is_alive():
            # Don't join - could block, and daemon thread will terminate anyway
            logger.info("Training thread will terminate")

# Start background preloading thread
def start_preloading():
    """Start preloading thread after app initialization"""
    def preload_in_background():
        try:
            logger.info("Starting background preloading of critical modules...")
            # Preload prediction models first as they're most used
            get_module('prediction_models')
            logger.info("Preloading completed!")
        except Exception as e:
            logger.error(f"Error in preloading thread: {str(e)}")
            traceback.print_exc()
    
    thread = threading.Thread(target=preload_in_background)
    thread.daemon = True
    thread.start()
    logger.info("Preload thread started")

# Enhanced module loading system with thread safety
def get_module(name):
    """Thread-safe lazy loading of modules with caching"""
    if name in _modules:
        return _modules[name]
    
    with _modules_lock:
        # Check again inside the lock
        if name in _modules:
            return _modules[name]
            
        try:
            start_time = time.time()
            
            if name == 'anthropic':
                import anthropic
                api_key = os.getenv('ANTHROPIC_API_KEY')
                if not api_key:
                    logger.warning("Anthropic API key not found")
                    api_key = "dummy_key_for_initialization"
                _modules[name] = anthropic.Anthropic(api_key=api_key)
                
            elif name == 'forecaster':
                from models.forecasting import RevenueForecast
                _modules[name] = RevenueForecast()
                
            elif name == 'teaching_assistant':
                from models.teaching_assistant import ProjectManagementTA
                _modules[name] = ProjectManagementTA()
                
            elif name == 'agent':
                from models.lake import FrozenLake
                _modules[name] = FrozenLake()
                
            elif name == 'training_manager':
                _modules[name] = TrainingManager()
                
            elif name == 'prediction_models':
                from models.predictor import model_info, enhanced_df, load_models
                model_info, enhanced_df = load_models()
                _modules['model_info'] = model_info
                _modules['enhanced_df'] = enhanced_df
                _modules[name] = True  # Mark as loaded
                
            load_time = time.time() - start_time
            logger.info(f"Module '{name}' loaded in {load_time:.2f} seconds")
            return _modules.get(name)
            
        except Exception as e:
            logger.error(f"Error loading module '{name}': {str(e)}")
            traceback.print_exc()
            return None

# Implement a simple in-memory cache
cache = {}
def cache_with_timeout(key, value, timeout=300):  # 5 minutes default
    cache[key] = {
        'value': value,
        'expires': time.time() + timeout
    }

def get_cached(key):
    if key in cache:
        data = cache[key]
        if time.time() < data['expires']:
            return data['value']
        else:
            del cache[key]  # Clear expired entry
    return None

# Basic page routes - these don't need the complex models
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/Liveconnect')
def liveconnect():
    return render_template('Liveconnect.html')

@app.route('/maplab')
def maplab():
    return render_template('maplab.html')

@app.route('/peas')
def peas():
    return render_template('Peas.html')

@app.route('/mba')
def mba():
    return render_template('mba.html')

@app.route('/voc')
def voc():
    return render_template('voc.html')

@app.route('/reinforcement_learning')
def rl():
    return render_template('reinforcement_learning.html')

@app.route('/teaching_assistant')
def teaching_assistant():
    return render_template('teaching_assistant.html')

@app.route('/cfbpredictions')
def predictions():
    return render_template('cfbpredictions.html')

@app.route('/teaching_feedback')
def teaching_feedback():
    return render_template('teaching_feedback.html')

@app.route('/forecasting')
def forecasting():
    return render_template('forecasting.html')

@app.route('/api/port')
def get_port():
    return jsonify({"port": request.host.split(':')[1] if ':' in request.host else "8080"})

@app.route('/health')
def health_check():
    return jsonify({"status": "ok", "timestamp": time.time()})

# Add REST endpoint to get current training status
@app.route('/api/training_status', methods=['GET'])
def get_training_status():
    """REST endpoint to get current training status"""
    try:
        agent = get_module('agent')
        if not agent:
            return jsonify({"error": "Agent not initialized"}), 503
            
        training_manager = get_module('training_manager')
        is_training = training_manager and training_manager.is_training
            
        metrics = agent.get_metrics()
        frame = agent.get_frame()
        
        return jsonify({
            "training_active": is_training,
            "metrics": metrics,
            "frame": frame,
            "server_time": time.time()
        })
    except Exception as e:
        logger.error(f"Error getting training status: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# API routes with improved loading
@app.route('/generate_colors', methods=['POST'])
def generate_colors():
    try:
        data = request.json
        if not data or "message" not in data:
            return jsonify({"error": "Invalid input: 'message' field is required."}), 400

        msg = data.get("message", "").strip()
        if not msg:
            return jsonify({"error": "Input 'message' cannot be empty."}), 400

        # Check cache first
        cache_key = f"colors_{msg}"
        cached_result = get_cached(cache_key)
        if cached_result:
            return jsonify({"colors": cached_result})

        # Get Anthropic client
        import anthropic
        client = get_module('anthropic')
        if not client:
            return jsonify({"error": "AI service not available"}), 503

        prompt = f"""
        You are a color palette generating assistant. Create a harmonious color palette based on this theme: {msg}
        Rules:
        1. Return ONLY hex color codes
        2. Provide 2-8 colors
        3. Format each color as: #RRGGBB
        4. Colors should work well together visually
        Example output: #FF5733 #33FF57 #5733FF
        """

        try:
            # Set a timeout for the API call
            response = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                messages=[
                    {"role": "user", "content": prompt}
                ],                
                max_tokens=100,
                temperature=0.7,
                timeout=15  # Reduced timeout
            )
            
            # Extract content from Anthropic's response structure
            assistant_reply = response.content[0].text.strip()
            colors = re.findall(r"#(?:[0-9a-fA-F]{6})", assistant_reply)
            
            if not colors:
                return jsonify({"error": "No valid colors found in the API response."}), 500

            # Cache the result for 1 day (86400 seconds)
            cache_with_timeout(cache_key, colors, 86400)
            return jsonify({"colors": colors})
            
        except anthropic.APITimeoutError:
            return jsonify({"error": "The request to the AI service timed out. Please try again."}), 504
        except anthropic.APIError as api_err:
            return jsonify({"error": f"AI service error: {str(api_err)}"}), 502
        
    except Exception as e:
        logger.error(f"Error in /generate_colors: {e}")
        traceback.print_exc()
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

# CFB Predictor routes
@app.route('/api/teams')
def get_teams():
    try:
        # Check cache first
        cached_teams = get_cached("cfb_teams")
        if cached_teams:
            return jsonify(cached_teams)
            
        # Ensure prediction models are loaded
        get_module('prediction_models')
        enhanced_df = _modules.get('enhanced_df')
        if enhanced_df is None:
            return jsonify({"error": "Team data not properly initialized"}), 500
            
        teams = []
        for _, row in enhanced_df.drop_duplicates(['homeTeam', 'home_conference']).iterrows():
            teams.append(f"{row['home_conference']} - {row['homeTeam']}")
        
        sorted_teams = sorted(set(teams))
        # Cache for 1 day
        cache_with_timeout("cfb_teams", sorted_teams, 86400)
        return jsonify(sorted_teams)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        # Ensure prediction models are loaded
        get_module('prediction_models')
        enhanced_df = _modules.get('enhanced_df')
        if enhanced_df is None:
            return jsonify({"error": "Prediction model not initialized"}), 503
            
        data = request.get_json()
        if not data or 'homeTeam' not in data or 'awayTeam' not in data:
            return jsonify({"error": "Missing team information"}), 400

        home_team = data['homeTeam'].split(' - ')[-1]
        away_team = data['awayTeam'].split(' - ')[-1]
        
        # Check cache
        cache_key = f"prediction_{home_team}_{away_team}"
        cached_result = get_cached(cache_key)
        if cached_result:
            return jsonify(cached_result)
        
        available_home_teams = enhanced_df['homeTeam'].unique()
        available_away_teams = enhanced_df['awayTeam'].unique()
        
        if home_team not in available_home_teams:
            return jsonify({"error": f"Invalid home team: {home_team}"}), 400
        if away_team not in available_away_teams:
            return jsonify({"error": f"Invalid away team: {away_team}"}), 400

        from models.predictor import predict_game_score
        prediction = predict_game_score(home_team, away_team, enhanced_df, 2024)
        
        if prediction:
            home_score, away_score, details = prediction
            
            response_data = {
                'homeTeam': {
                    'name': home_team,
                    'predictedScore': home_score,
                    'avgPoints': float(enhanced_df[enhanced_df['homeTeam'] == home_team]['homePoints'].mean()),
                    'winPercentage': None
                },
                'awayTeam': {
                    'name': away_team,
                    'predictedScore': away_score,
                    'avgPoints': float(enhanced_df[enhanced_df['awayTeam'] == away_team]['awayPoints'].mean()),
                    'winPercentage': None
                },
                'prediction': {
                    'spread': abs(home_score - away_score),
                    'total': home_score + away_score,
                    'favorite': home_team if home_score > away_score else away_team,
                    'underdog': away_team if home_score > away_score else home_team,
                    'factors': details.get('factors', {}),
                    'weights': details.get('weights', {})
                }
            }
            
            # Cache for 1 day
            cache_with_timeout(cache_key, response_data, 86400)
            return jsonify(response_data)
        else:
            return jsonify({"error": "Unable to generate prediction"}), 500

    except Exception as e:
        logger.error("Error in /api/predict:")
        logger.error(str(e))
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/model-info')
def get_model_info():
    try:
        # Check cache first
        cached_info = get_cached("model_info")
        if cached_info:
            return jsonify(cached_info)
            
        # Ensure prediction models are loaded
        get_module('prediction_models')
        model_info = _modules.get('model_info')
        
        result = {
            'homeModel': {
                'metrics': model_info['models']['homePoints']['metrics']
            },
            'awayModel': {
                'metrics': model_info['models']['awayPoints']['metrics']
            }
        }
        
        # Cache for 1 day
        cache_with_timeout("model_info", result, 86400)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Teaching Assistant routes
@app.route('/api/analyze-post', methods=['POST'])
def analyze_post():
    try:
        teaching_assistant = get_module('teaching_assistant')
        if not teaching_assistant:
            return jsonify({"error": "Teaching assistant not initialized"}), 503
            
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
            
        week = data.get('week')
        discussion = data.get('discussion')
        post = data.get('post', '').strip()
        
        if not all([week, discussion, post]):
            return jsonify({"error": "Missing required fields"}), 400
            
        # Check cache for identical posts
        cache_key = f"post_analysis_{week}_{discussion}_{hash(post)}"
        cached_result = get_cached(cache_key)
        if cached_result:
            return jsonify(cached_result)
            
        feedback = teaching_assistant.analyze_post(week, discussion, post)
        
        # Cache for 1 hour
        cache_with_timeout(cache_key, feedback, 3600)
        return jsonify(feedback)
            
    except Exception as e:
        logger.error(f"Error in analyze_post route: {str(e)}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/week-content/<int:week>')
def get_week_content(week):
    try:
        # Check cache first
        cache_key = f"week_content_{week}"
        cached_content = get_cached(cache_key)
        if cached_content:
            return jsonify(cached_content)
            
        teaching_assistant = get_module('teaching_assistant')
        if not teaching_assistant:
            return jsonify({"error": "Teaching assistant not initialized"}), 503
            
        content = teaching_assistant.get_week_content(week)
        
        # Cache for 1 day (content is static)
        cache_with_timeout(cache_key, content, 86400)
        return jsonify(content)
    except Exception as e:
        logger.error(f"Error getting week content: {str(e)}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# Reinforcement Learning routes
@app.route('/start_training', methods=['POST'])
def start_training():
    try:
        agent = get_module('agent')
        training_manager = get_module('training_manager')
        if not agent or not training_manager:
            return jsonify({"error": "Failed to initialize agent"}), 500

        data = request.get_json()
        learning_rate = float(data.get('learning_rate', 0.001))
        discount_factor = float(data.get('discount_factor', 0.99))
        exploration_rate = float(data.get('exploration_rate', 0.1))
        
        # Explicitly reset training complete flag
        agent.training_complete = False
        
        agent.update_parameters(
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            epsilon=exploration_rate
        )
        
        training_manager.start_training(agent)
        return jsonify({"status": "success"})
        
    except Exception as e:
        logger.error(f"Error starting training: {str(e)}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/stop_training', methods=['POST'])
def stop_training():
    try:
        training_manager = get_module('training_manager')
        if training_manager:
            training_manager.stop_training()
        return jsonify({"status": "success"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/randomize_map', methods=['POST'])
def randomize_map():
    try:
        agent = get_module('agent')
        if not agent:
            return jsonify({"error": "Failed to initialize agent"}), 500
            
        data = request.get_json()
        size = int(data.get('size', 8))
            
        success = agent.randomize_map(size)
        if not success:
            return jsonify({"error": "Failed to randomize map"}), 500
            
        metrics = agent.get_metrics()
        frame = agent.get_frame()
            
        return jsonify({
            "status": "success",
            "metrics": metrics,
            "frame": frame
        })
        
    except Exception as e:
        logger.error(f"Error in randomize_map: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/get_state', methods=['GET'])
def get_state():
    try:
        agent = get_module('agent')
        if not agent:
            return jsonify({"error": "Agent not initialized"}), 500
            
        metrics = agent.get_metrics()
        frame = agent.get_frame()
            
        return jsonify({
            "metrics": metrics,
            "frame": frame,
            "server_time": time.time()
        })
        
    except Exception as e:
        logger.error(f"Error in get_state: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/save_model', methods=['POST'])
def save_model():
    try:
        agent = get_module('agent')
        if agent:
            agent.save_model()
        return jsonify({"status": "success"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/load_model', methods=['POST'])
def load_model():
    try:
        agent = get_module('agent')
        if agent:
            agent.load_model()
        return jsonify({"status": "success"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
# Forecasting Routes with caching
@app.route('/api/forecast/internal', methods=['POST'])
def handle_internal_forecast():
    try:
        forecaster = get_module('forecaster')
        if not forecaster:
            return jsonify({"error": "Forecasting module not initialized"}), 503
        return forecaster.handle_internal_forecast()
    except Exception as e:
        logger.error(f"Error in internal forecast: {str(e)}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/forecast/upsell', methods=['POST'])
def handle_upsell_forecast():
    try:
        forecaster = get_module('forecaster')
        if not forecaster:
            return jsonify({"error": "Forecasting module not initialized"}), 503
        return forecaster.handle_upsell_forecast()
    except Exception as e:
        logger.error(f"Error in upsell forecast: {str(e)}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/forecast/market', methods=['POST'])
def handle_market_forecast():
    try:
        forecaster = get_module('forecaster')
        if not forecaster:
            return jsonify({"error": "Forecasting module not initialized"}), 503
        return forecaster.handle_market_forecast()
    except Exception as e:
        logger.error(f"Error in market forecast: {str(e)}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/forecast/satisfaction', methods=['POST'])
def handle_satisfaction_forecast():
    try:
        forecaster = get_module('forecaster')
        if not forecaster:
            return jsonify({"error": "Forecasting module not initialized"}), 503
        return forecaster.handle_satisfaction_forecast()
    except Exception as e:
        logger.error(f"Error in satisfaction forecast: {str(e)}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

def find_available_port(start_port=5000, max_attempts=100):
    """Find an available port starting from start_port"""
    for port in range(start_port, start_port + max_attempts):
        if not is_port_in_use(port):
            return port
    raise RuntimeError(f"No available ports found between {start_port} and {start_port + max_attempts}")

def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(('0.0.0.0', port))
            return False
        except OSError:
            return True

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    logger.info(f"Starting app on port {port}")
    
    # Start preloading critical modules in background thread
    start_preloading()
    
    # Start the Flask app
    app.run(host='0.0.0.0', port=port)