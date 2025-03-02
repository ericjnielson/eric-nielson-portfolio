"""
Main Flask application file for portfolio website with various features including:
- Color generation
- CFB predictions
- Teaching assistant
- Reinforcement learning
- Revenue forecasting
"""

# *** IMPORTANT: Monkey patch eventlet before all other imports ***
import eventlet
eventlet.monkey_patch()

# Standard library imports - only import what's absolutely necessary
import os
import re
import json
import base64
import socket
import traceback
from typing import Dict, List
import time

# Initialize empty values for lazy loading
FORECASTER = None
AGENT = None
TRAINING_MANAGER = None
TEACHING_ASSISTANT = None
MODEL_INFO = None
ENHANCED_DF = None
ANTHROPIC_CLIENT = None

# Create the Flask app instance
from flask import Flask, request, render_template, jsonify, url_for
app = Flask(__name__, 
            static_folder='static', 
            static_url_path='/static',
            template_folder='templates')

# Configure app with minimal settings
app.config['SECRET_KEY'] = os.urandom(24)
# Make sure url_for() works correctly by configuring the application
app.config['SERVER_NAME'] = os.environ.get('SERVER_NAME', None)
app.config['PREFERRED_URL_SCHEME'] = 'https'
app.config['APPLICATION_ROOT'] = '/'
# Set up CORS - Light version
from flask_cors import CORS
CORS(app)

# Initialize Socket.IO with optimized settings
from flask_socketio import SocketIO, emit
socketio = SocketIO(
    app,
    path='/ws/socket.io',
    async_mode='eventlet',
    cors_allowed_origins="*",
    logger=False,  # Disable logging for production
    engineio_logger=False,  # Disable engineio logging
    ping_timeout=60,
    ping_interval=25
)

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

def lazy_load_anthropic():
    """Lazy load Anthropic client"""
    global ANTHROPIC_CLIENT
    if ANTHROPIC_CLIENT is None:
        try:
            import anthropic
            api_key = os.getenv('ANTHROPIC_API_KEY')
            if not api_key:
                print("Warning: Anthropic API key not found")
                api_key = "dummy_key_for_initialization"
            ANTHROPIC_CLIENT = anthropic.Anthropic(api_key=api_key)
            print("Successfully initialized Anthropic client")
        except Exception as e:
            print(f"Error initializing Anthropic client: {str(e)}")
            ANTHROPIC_CLIENT = None
    return ANTHROPIC_CLIENT

def lazy_load_forecaster():
    """Lazy load RevenueForecast"""
    global FORECASTER
    if FORECASTER is None:
        try:
            from models.forecasting import RevenueForecast
            FORECASTER = RevenueForecast()
            print("Successfully initialized forecaster")
        except Exception as e:
            print(f"Error initializing forecaster: {str(e)}")
            FORECASTER = None
    return FORECASTER

def lazy_load_teaching_assistant():
    """Lazy load teaching assistant"""
    global TEACHING_ASSISTANT
    if TEACHING_ASSISTANT is None:
        try:
            from models.teaching_assistant import ProjectManagementTA
            TEACHING_ASSISTANT = ProjectManagementTA()
            print("Successfully initialized teaching assistant")
        except Exception as e:
            print(f"Error initializing teaching assistant: {str(e)}")
            traceback.print_exc()
            TEACHING_ASSISTANT = None
    return TEACHING_ASSISTANT

def lazy_load_agent():
    """Lazy load reinforcement learning agent"""
    global AGENT, TRAINING_MANAGER
    if AGENT is None:
        try:
            from models.lake import FrozenLake
            from models.training_manager import TrainingManager
            AGENT = FrozenLake()
            TRAINING_MANAGER = TrainingManager(socketio)
            print("Successfully initialized Frozen Lake agent")
        except Exception as e:
            print(f"Error initializing agent: {str(e)}")
            traceback.print_exc()
            AGENT = None
            TRAINING_MANAGER = None
    return AGENT, TRAINING_MANAGER

def lazy_load_prediction_models():
    """Lazy load prediction models"""
    global MODEL_INFO, ENHANCED_DF
    if MODEL_INFO is None:
        try:
            from models.predictor import model_info, enhanced_df, load_models
            MODEL_INFO, ENHANCED_DF = load_models()
            print("Successfully loaded prediction models and data")
        except Exception as e:
            print(f"Error loading prediction models: {str(e)}")
            traceback.print_exc()
            MODEL_INFO = {}
            ENHANCED_DF = None
    return MODEL_INFO, ENHANCED_DF

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

# API routes with lazy loading
@app.route('/generate_colors', methods=['POST'])
def generate_colors():
    try:
        data = request.json
        if not data or "message" not in data:
            return jsonify({"error": "Invalid input: 'message' field is required."}), 400

        msg = data.get("message", "").strip()
        if not msg:
            return jsonify({"error": "Input 'message' cannot be empty."}), 400

        # Lazy load Anthropic client
        client = lazy_load_anthropic()
        if not client:
            return jsonify({"error": "AI service not available"}), 503

        import anthropic
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
                max_tokens=1024,
                temperature=0.7,
                timeout=20  # 20-second timeout for the API call
            )
            
            # Extract content from Anthropic's response structure
            assistant_reply = response.content[0].text.strip()
            colors = re.findall(r"#(?:[0-9a-fA-F]{6})", assistant_reply)
            
            if not colors:
                return jsonify({"error": "No valid colors found in the API response."}), 500

            return jsonify({"colors": colors})
            
        except anthropic.APITimeoutError:
            return jsonify({"error": "The request to the AI service timed out. Please try again."}), 504
        except anthropic.APIError as api_err:
            return jsonify({"error": f"AI service error: {str(api_err)}"}), 502
        
    except Exception as e:
        print(f"Error in /generate_colors: {e}")
        traceback.print_exc()
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

# CFB Predictor routes
@app.route('/api/teams')
def get_teams():
    try:
        _, enhanced_df = lazy_load_prediction_models()
        if enhanced_df is None:
            return jsonify({"error": "Team data not properly initialized"}), 500
            
        teams = []
        for _, row in enhanced_df.drop_duplicates(['homeTeam', 'home_conference']).iterrows():
            teams.append(f"{row['home_conference']} - {row['homeTeam']}")
        return jsonify(sorted(set(teams)))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        _, enhanced_df = lazy_load_prediction_models()
        if enhanced_df is None:
            return jsonify({"error": "Prediction model not initialized"}), 503
            
        data = request.get_json()
        if not data or 'homeTeam' not in data or 'awayTeam' not in data:
            return jsonify({"error": "Missing team information"}), 400

        home_team = data['homeTeam'].split(' - ')[-1]
        away_team = data['awayTeam'].split(' - ')[-1]
        
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
            return jsonify(response_data)
        else:
            return jsonify({"error": "Unable to generate prediction"}), 500

    except Exception as e:
        print("Error in /api/predict:")
        print(str(e))
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route('/api/model-info')
def get_model_info():
    try:
        model_info, _ = lazy_load_prediction_models()
        return jsonify({
            'homeModel': {
                'metrics': model_info['models']['homePoints']['metrics']
            },
            'awayModel': {
                'metrics': model_info['models']['awayPoints']['metrics']
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Teaching Assistant routes
@app.route('/api/analyze-post', methods=['POST'])
def analyze_post():
    try:
        teaching_assistant = lazy_load_teaching_assistant()
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
            
        feedback = teaching_assistant.analyze_post(week, discussion, post)
        return jsonify(feedback)
            
    except Exception as e:
        print(f"Error in analyze_post route: {str(e)}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/week-content/<int:week>')
def get_week_content(week):
    try:
        teaching_assistant = lazy_load_teaching_assistant()
        if not teaching_assistant:
            return jsonify({"error": "Teaching assistant not initialized"}), 503
            
        content = teaching_assistant.get_week_content(week)
        return jsonify(content)
    except Exception as e:
        print(f"Error getting week content: {str(e)}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# Reinforcement Learning routes
@app.route('/start_training', methods=['POST'])
def start_training():
    try:
        agent, training_manager = lazy_load_agent()
        if not agent:
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
        print(f"Error starting training: {str(e)}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/stop_training', methods=['POST'])
def stop_training():
    try:
        _, training_manager = lazy_load_agent()
        if training_manager:
            training_manager.stop_training()
        return jsonify({"status": "success"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/randomize_map', methods=['POST'])
def randomize_map():
    try:
        agent, _ = lazy_load_agent()
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
        print(f"Error in randomize_map: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/get_state', methods=['GET'])
def get_state():
    try:
        agent, _ = lazy_load_agent()
        if not agent:
            return jsonify({"error": "Agent not initialized"}), 500
            
        metrics = agent.get_metrics()
        frame = agent.get_frame()
            
        return jsonify({
            "metrics": metrics,
            "frame": frame
        })
        
    except Exception as e:
        print(f"Error in get_state: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/save_model', methods=['POST'])
def save_model():
    try:
        agent, _ = lazy_load_agent()
        if agent:
            agent.save_model()
        return jsonify({"status": "success"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/load_model', methods=['POST'])
def load_model():
    try:
        agent, _ = lazy_load_agent()
        if agent:
            agent.load_model()
        return jsonify({"status": "success"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
# Forecasting Routes
@app.route('/api/forecast/internal', methods=['POST'])
def handle_internal_forecast():
    try:
        forecaster = lazy_load_forecaster()
        if not forecaster:
            return jsonify({"error": "Forecasting module not initialized"}), 503
        return forecaster.handle_internal_forecast()
    except Exception as e:
        print(f"Error in internal forecast: {str(e)}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/forecast/upsell', methods=['POST'])
def handle_upsell_forecast():
    try:
        forecaster = lazy_load_forecaster()
        if not forecaster:
            return jsonify({"error": "Forecasting module not initialized"}), 503
        return forecaster.handle_upsell_forecast()
    except Exception as e:
        print(f"Error in upsell forecast: {str(e)}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/forecast/market', methods=['POST'])
def handle_market_forecast():
    try:
        forecaster = lazy_load_forecaster()
        if not forecaster:
            return jsonify({"error": "Forecasting module not initialized"}), 503
        return forecaster.handle_market_forecast()
    except Exception as e:
        print(f"Error in market forecast: {str(e)}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/forecast/satisfaction', methods=['POST'])
def handle_satisfaction_forecast():
    try:
        forecaster = lazy_load_forecaster()
        if not forecaster:
            return jsonify({"error": "Forecasting module not initialized"}), 503
        return forecaster.handle_satisfaction_forecast()
    except Exception as e:
        print(f"Error in satisfaction forecast: {str(e)}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# Socket.IO event handlers
@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')
    agent, training_manager = lazy_load_agent()
    if training_manager and training_manager.is_training:
        training_manager.stop_training()

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

def run_app(port):
    try:
        # Set buffer size for frame data
        eventlet.wsgi.MAX_BUFFER_SIZE = 16777216  # 16MB
        
        print(f"Starting server on port: {port}")
        socketio.run(
            app,
            host='0.0.0.0',
            port=port,
            log_output=True,
            use_reloader=False,
            debug=False,  # Disable debug mode for production
            allow_unsafe_werkzeug=True
        )
    except Exception as e:
        print(f"Error starting server: {e}")
        traceback.print_exc()

if __name__ == '__main__':
    try:
        port = int(os.environ.get("PORT", 8080))
        print(f"Using port: {port}")
        run_app(port)
    except Exception as e:
        print(f"Error starting server: {e}")
        traceback.print_exc()