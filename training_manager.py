from flask_socketio import SocketIO, emit
import threading
import time
import base64
import cv2

class TrainingManager:
    def __init__(self, socketio):
        self.is_training = False
        self.training_thread = None
        self.socketio = socketio
        
    def start_training(self, agent):
        """Start training loop in a separate thread"""
        def training_loop():
            while self.is_training:
                try:
                    # Check if training is complete
                    if agent.training_complete:
                        print("Training complete! Stopping training loop.")
                        self.is_training = False
                        self.socketio.emit('training_complete', {
                            'message': 'Training complete!',
                            'metrics': agent.get_metrics()
                        })
                        break

                    # Run training episode
                    reward = agent.train_episode()
                    metrics = agent.get_metrics()
                    frame = agent.get_frame()
                    
                    # Prepare frame for emission
                    frame_b64 = None
                    if frame is not None:
                        _, buffer = cv2.imencode('.jpg', frame)
                        frame_b64 = base64.b64encode(buffer).decode('utf-8')
                    
                    # Emit update to client
                    self.socketio.emit('training_update', {
                        'frame': frame_b64,
                        'metrics': metrics,
                        'reward': float(reward)
                    })

                    # Check if max episodes reached or success rate achieved
                    if metrics['training_complete']:
                        print("Training complete! Stopping training loop.")
                        self.is_training = False
                        self.socketio.emit('training_complete', {
                            'message': 'Training complete!',
                            'metrics': metrics
                        })
                        break
                    
                    # Control update frequency
                    time.sleep(0.1)
                    
                except Exception as e:
                    print(f"Error in training loop: {e}")
                    self.is_training = False
                    self.socketio.emit('training_error', {'error': str(e)})
                    break
                
        self.is_training = True
        self.training_thread = threading.Thread(target=training_loop)
        self.training_thread.start()
        
    def stop_training(self):
        """Stop the training loop"""
        self.is_training = False
        if self.training_thread:
            self.training_thread.join()