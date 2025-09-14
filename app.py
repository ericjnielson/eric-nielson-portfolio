import eventlet
eventlet.monkey_patch()

# Standard library imports
import os
import re
import json
import base64
import socket
import traceback
from typing import Dict, List

# Third-party imports
import cv2
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit

# Local imports
from lake import FrozenLake
from training_manager import TrainingManager
from teaching_assistant import ProjectManagementTA
from forecasting import RevenueForecast
from models.predictor import model_info, predict_score, games_df

def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(('0.0.0.0', port))
            return False
        except OSError:
            return True

def find_available_port(start_port=5000, max_attempts=100):
    """Find an available port starting from start_port"""
    for port in range(start_port, start_port + max_attempts):
        if not is_port_in_use(port):
            return port
    raise RuntimeError(f"No available ports found between {start_port} and {start_port + max_attempts}")

class PortfolioApp:
    def __init__(self, flask_app=None):
        # Load environment variables
        load_dotenv()
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.heygen_api_key = os.getenv('HEYGEN_API_KEY')

        if not self.openai_api_key:
            raise ValueError("OpenAI API key not found in environment variables")

        # Initialize Flask and extensions
        self.app = flask_app if flask_app else Flask(__name__)
        self.setup_cors()
        self.app.secret_key = os.urandom(24)
        
        # Initialize SocketIO
        self.socketio = SocketIO(
            self.app,
            async_mode='eventlet',
            cors_allowed_origins="*",
            logger=True,
            engineio_logger=True
        )

        # Initialize components
        self.client = OpenAI(api_key=self.openai_api_key)
        self.forecaster = RevenueForecast()
        self.teaching_assistant = self.initialize_teaching_assistant()
        
        # Initialize RL components
        self.agent = None
        self.training_manager = None
        self.initialize_agent()

        # Register routes
        self.register_routes()
        self.register_socketio_handlers()

    def setup_cors(self):
        """Configure CORS for the application"""
        CORS(self.app, resources={
            r"/*": {
                "origins": "*",
                "allow_headers": "*",
                "expose_headers": "*",
                "methods": ["GET", "POST", "OPTIONS"],
                "supports_credentials": True
            }
        })

    def initialize_teaching_assistant(self):
        """Initialize the teaching assistant component"""
        try:
            assistant = ProjectManagementTA()
            print("Successfully initialized teaching assistant")
            return assistant
        except Exception as e:
            print(f"Error initializing teaching assistant: {str(e)}")
            traceback.print_exc()
            raise

    def initialize_agent(self):
        """Initialize the reinforcement learning agent"""
        try:
            self.agent = FrozenLake()
            self.training_manager = TrainingManager(self.socketio)
            print("Successfully initialized Frozen Lake agent")
            return True
        except Exception as e:
            print(f"Error initializing agent: {str(e)}")
            return False

    def register_routes(self):
        """Register all route handlers"""

        @self.app.route('/api/port')
        def get_port():
            return jsonify({"port": request.host.split(':')[1]})
        # Basic page routes
        @self.app.route('/')
        def index():
            return self.index()

        @self.app.route('/Liveconnect')
        def liveconnect():
            return self.liveconnect()

        @self.app.route('/maplab')
        def maplab():
            return self.maplab()

        @self.app.route('/peas')
        def peas():
            return self.peas()

        @self.app.route('/mba')
        def mba():
            return self.mba()

        @self.app.route('/voc')
        def voc():
            return self.voc()

        @self.app.route('/reinforcement_learning')
        def rl():
            return self.rl()

        @self.app.route('/teaching_assistant')
        def teaching_assistant():
            return self.teaching_assistant_page()

        @self.app.route('/cfbpredictions')
        def predictions():
            return self.predictions()

        @self.app.route('/forecasting')
        def forecasting():
            return self.forecasting()

        # API routes
        # Color Generator
        @self.app.route('/generate_colors', methods=['POST'])
        def generate_colors():
            return self.generate_colors()
        
        # CFB Predictor
        @self.app.route('/api/teams')
        def get_teams():
            return self.get_teams()

        @self.app.route('/api/predict', methods=['POST'])
        def predict():
            return self.predict()

        @self.app.route('/api/model-info')
        def get_model_info():
            return self.get_model_info()
        
        # Teaching Assistant
        @self.app.route('/api/analyze-post', methods=['POST'])
        def analyze_post():
            return self.analyze_post()

        @self.app.route('/api/week-content/<int:week>')
        def get_week_content(week):
            return self.get_week_content(week)
        
        # Reinforcement Learning
        @self.app.route('/start_training', methods=['POST'])
        def start_training():
            return self.start_training()

        @self.app.route('/stop_training', methods=['POST'])
        def stop_training():
            return self.stop_training()

        @self.app.route('/train_step', methods=['POST'])
        def train_step():
            return self.train_step()

        @self.app.route('/get_state', methods=['GET'])
        def get_state():
            return self.get_state()

        @self.app.route('/save_model', methods=['POST'])
        def save_model():
            return self.save_model()

        @self.app.route('/load_model', methods=['POST'])
        def load_model():
            return self.load_model()
        
        @self.app.route('/randomize_map', methods=['POST'])
        def randomize_map():
            return self.randomize_map()
        # Forecasting Routes
        forecast_routes = [
            ('/api/forecast/internal', 'internal_forecast', self.handle_internal_forecast),
            ('/api/forecast/upsell', 'upsell_forecast', self.handle_upsell_forecast),
            ('/api/forecast/market', 'market_forecast', self.handle_market_forecast),
            ('/api/forecast/satisfaction', 'satisfaction_forecast', self.handle_satisfaction_forecast)
        ]
        
        for route, endpoint, handler in forecast_routes:
            self.app.add_url_rule(route, endpoint, handler, methods=['POST'])

    def register_socketio_handlers(self):
        """Register SocketIO event handlers"""
        @self.socketio.on('connect')
        def handle_connect():
            print('Client connected')
            if self.agent is None:
                self.initialize_agent()

        @self.socketio.on('disconnect')
        def handle_disconnect():
            print('Client disconnected')
            if self.training_manager and self.training_manager.is_training:
                self.training_manager.stop_training()

    ### Route Handlers ###

    # Basic page handlers
    def index(self):
        return render_template('index.html')

    def liveconnect(self):
        return render_template('Liveconnect.html')

    def maplab(self):
        return render_template('Maplab.html')

    def peas(self):
        return render_template('Peas.html')

    def mba(self):
        return render_template('mba.html')

    def voc(self):
        return render_template('voc.html')

    def forecasting(self):
        return render_template('forecasting.html')

    def rl(self):
        return render_template('reinforcement_learning.html')

    def teaching_assistant_page(self):
        return render_template('teaching_assistant.html')

    def predictions(self):
        return render_template('cfbpredictions.html')

    # Color Generator handlers
    def generate_colors(self):
        try:
            data = request.json
            print(f"Received data: {data}")

            if not data or "message" not in data:
                raise ValueError("Invalid input: 'message' field is required.")

            msg = data.get("message", "").strip()
            if not msg:
                raise ValueError("Input 'message' cannot be empty.")

            prompt = f"""
            You are a color palette generating assistant. Create a harmonious color palette based on this theme: {msg}
            Rules:
            1. Return ONLY hex color codes
            2. Provide 2-8 colors
            3. Format each color as: #RRGGBB
            4. Colors should work well together visually
            Example output: #FF5733 #33FF57 #5733FF
            """

            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=0.7
            )

            assistant_reply = response.choices[0].message.content.strip()
            colors = re.findall(r"#(?:[0-9a-fA-F]{6})", assistant_reply)

            if not colors:
                raise ValueError("No valid colors found in the OpenAI response.")

            return jsonify({"colors": colors})

        except Exception as e:
            print(f"Error in /generate_colors: {e}")
            return jsonify({"error": f"An error occurred: {str(e)}"}), 500

    # CFB Predictor handlers
    def get_teams(self):
        try:
            teams = []
            for _, row in games_df.drop_duplicates(['homeTeam', 'home_conference']).iterrows():
                teams.append(f"{row['home_conference']} - {row['homeTeam']}")
            return jsonify(sorted(set(teams)))
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    def predict(self):
        try:
            data = request.json
            if not data or 'homeTeam' not in data or 'awayTeam' not in data:
                return jsonify({"error": "Missing team information"}), 400

            home_team = data['homeTeam'].split(' - ')[-1]
            away_team = data['awayTeam'].split(' - ')[-1]
            
            available_home_teams = games_df['homeTeam'].unique()
            available_away_teams = games_df['awayTeam'].unique()
            
            if home_team not in available_home_teams:
                return jsonify({"error": f"Invalid home team: {home_team}"}), 400
            if away_team not in available_away_teams:
                return jsonify({"error": f"Invalid away team: {away_team}"}), 400

            prediction = predict_score(home_team, away_team, games_df, 2024)
            
            if prediction:
                response_data = {
                    'homeTeam': {
                        'name': home_team,
                        'predictedScore': prediction['scores']['home']['score'],
                        'avgPoints': prediction['scores']['home']['stats']['avg_score'],
                        'winPercentage': None
                    },
                    'awayTeam': {
                        'name': away_team,
                        'predictedScore': prediction['scores']['away']['score'],
                        'avgPoints': prediction['scores']['away']['stats']['avg_score'],
                        'winPercentage': None
                    },
                    'prediction': {
                        'spread': prediction['prediction']['spread'],
                        'total': prediction['prediction']['total'],
                        'favorite': prediction['prediction']['favorite'],
                        'underdog': prediction['prediction']['underdog'],
                        'factors': prediction['factors'],
                        'weights': prediction['weights']
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

    def get_model_info(self):
        try:
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

    # Teaching Assistant handlers
    def analyze_post(self):
        try:
            data = request.get_json()
            if not data:
                return jsonify({"error": "No data provided"}), 400
                
            week = data.get('week')
            discussion = data.get('discussion')
            post = data.get('post', '').strip()
            
            if not all([week, discussion, post]):
                return jsonify({"error": "Missing required fields"}), 400
                
            feedback = self.teaching_assistant.analyze_post(week, discussion, post)
            return jsonify(feedback)
                
        except Exception as e:
            print(f"Error in analyze_post route: {str(e)}")
            traceback.print_exc()
            return jsonify({"error": str(e)}), 500

    def get_week_content(self, week):
        try:
            content = self.teaching_assistant.get_week_content(week)
            return jsonify(content)
        except Exception as e:
            print(f"Error getting week content: {str(e)}")
            traceback.print_exc()
            return jsonify({"error": str(e)}), 500

    # Reinforcement Learning handlers
    def start_training(self):
        try:
            if self.agent is None:
                if not self.initialize_agent():
                    return jsonify({"error": "Failed to initialize agent"}), 500

            data = request.get_json()
            learning_rate = float(data.get('learning_rate', 0.001))
            discount_factor = float(data.get('discount_factor', 0.99))
            exploration_rate = float(data.get('exploration_rate', 0.1))
            
            self.agent.update_parameters(
                learning_rate=learning_rate,
                discount_factor=discount_factor,
                epsilon=exploration_rate
            )
            
            self.training_manager.start_training(self.agent)
            return jsonify({"status": "success"})
            
        except Exception as e:
            print(f"Error starting training: {str(e)}")
            return jsonify({"error": str(e)}), 500

    def stop_training(self):
        try:
            if self.training_manager:
                self.training_manager.stop_training()
            return jsonify({"status": "success"})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    def randomize_map(self):
        try:
            data = request.get_json()
            size = int(data.get('size', 8))
            
            if self.agent is None:
                if not self.initialize_agent():
                    return jsonify({"error": "Failed to initialize agent"}), 500
                    
            self.agent.randomize_map(size)
            metrics = self.agent.get_metrics()
            frame = self.agent.get_frame()
            
            frame_b64 = None
            if frame is not None:
                _, buffer = cv2.imencode('.jpg', frame)
                frame_b64 = base64.b64encode(buffer).decode('utf-8')
                
            return jsonify({
                "status": "success",
                "metrics": metrics,
                "frame": frame_b64
            })
            
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    def train_step(self):
        try:
            if self.agent is None:
                if not self.initialize_agent():
                    return jsonify({
                        "error": "Failed to initialize agent",
                        "metrics": {
                            "average_reward": 0,
                            "episode_count": 0,
                            "success_rate": 0
                        }
                    }), 500

            data = request.get_json()
            if not data:
                return jsonify({"error": "No data provided"}), 400

            try:
                learning_rate = float(data.get('learning_rate', 0.001))
                discount_factor = float(data.get('discount_factor', 0.99))
                exploration_rate = float(data.get('exploration_rate', 0.1))
                
                self.agent.update_parameters(
                    learning_rate=learning_rate,
                    discount_factor=discount_factor,
                    epsilon=exploration_rate
                )
            except (TypeError, ValueError) as e:
                return jsonify({"error": f"Invalid parameters: {str(e)}"}), 400

            try:
                reward = self.agent.train_episode()
                metrics = self.agent.get_metrics()
                frame = self.agent.get_frame()
                
                frame_b64 = None
                if frame is not None and isinstance(frame, np.ndarray):
                    if frame.dtype != np.uint8:
                        frame = (frame * 255).astype(np.uint8)
                    
                    _, buffer = cv2.imencode('.jpg', frame)
                    frame_b64 = base64.b64encode(buffer).decode('utf-8')

                return jsonify({
                    'metrics': metrics,
                    'frame': frame_b64,
                    'reward': float(reward)
                })
                
            except Exception as e:
                print(f"Error during training: {str(e)}")
                traceback.print_exc()
                return jsonify({
                    "error": f"Training error: {str(e)}",
                    "metrics": self.agent.get_metrics() if self.agent else {
                        "average_reward": 0,
                        "episode_count": 0,
                        "success_rate": 0
                    }
                }), 500

        except Exception as e:
            print(f"Error in train_step route: {str(e)}")
            traceback.print_exc()
            return jsonify({
                "error": str(e),
                "metrics": {
                    "average_reward": 0,
                    "episode_count": 0,
                    "success_rate": 0
                }
            }), 500

    def get_state(self):
        try:
            if self.agent is None:
                if not self.initialize_agent():
                    return jsonify({"error": "Agent not initialized"}), 500
                    
            metrics = self.agent.get_metrics()
            frame = self.agent.get_frame()
            
            frame_b64 = None
            if frame is not None:
                _, buffer = cv2.imencode('.jpg', frame)
                frame_b64 = base64.b64encode(buffer).decode('utf-8')
                
            return jsonify({
                "metrics": metrics,
                "frame": frame_b64
            })
            
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    def save_model(self):
        try:
            if self.agent:
                self.agent.save_model()
            return jsonify({"status": "success"})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    def load_model(self):
        try:
            if self.agent:
                self.agent.load_model()
            return jsonify({"status": "success"})
        except Exception as e:
            return jsonify({"error": str(e)}), 500
        
    def handle_internal_forecast(self):
        return self.forecaster.handle_internal_forecast()

    def handle_upsell_forecast(self):
        return self.forecaster.handle_upsell_forecast()

    def handle_market_forecast(self):
        return self.forecaster.handle_market_forecast()

    def handle_satisfaction_forecast(self):
        return self.forecaster.handle_satisfaction_forecast()
        
def create_app():
    """Create and configure the application"""
    app = Flask(__name__)
    try:
        portfolio_app = PortfolioApp(app)
        return app, portfolio_app.socketio
    except Exception as e:
        print(f"Error creating application: {str(e)}")
        traceback.print_exc()
        raise


if __name__ == "__main__":
    import argparse

    # Prefer PORT from env (IDX sets this). Default to 8080 if missing.
    port = int(os.getenv("PORT", "8080"))
    host = "0.0.0.0"

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=host)
    parser.add_argument("--port", type=int, default=port)
    args = parser.parse_args()

    try:
        app, socketio = create_app()
        print(f"✅ Starting server on http://{args.host}:{args.port}")
        socketio.run(
            app,
            debug=True,
            host=args.host,
            port=args.port,
            allow_unsafe_werkzeug=True
        )
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
