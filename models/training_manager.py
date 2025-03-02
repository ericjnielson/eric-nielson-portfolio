from flask_socketio import SocketIO, emit
import threading
import time
import base64
from PIL import Image
import io
import traceback

class TrainingManager:
    def __init__(self, socketio):
        self.is_training = False
        self.training_thread = None
        self.socketio = socketio
        
    def start_training(self, agent):
        """Start training loop in a separate thread"""
        print("Starting training loop")
        agent.training_complete = False  # Reset training flag
        
        def training_loop():
            print("Training loop started")
            
            while self.is_training:
                try:
                    # Check if training is complete
                    if agent.training_complete:
                        print("Training complete! Stopping training loop.")
                        self.is_training = False
                        
                        # Get final metrics and frame
                        final_metrics = agent.get_metrics()
                        final_frame = agent.get_frame()
                        
                        # Send completion notification
                        self.socketio.emit('training_complete', {
                            'message': 'Training complete!',
                            'metrics': final_metrics,
                            'frame': final_frame
                        })
                        break

                    # Run training episode
                    reward = agent.train_episode()
                    metrics = agent.get_metrics()
                    frame = agent.get_frame()
                    
                    # Log progress
                    if metrics['episode_count'] % 10 == 0:
                        print(f"Episode {metrics['episode_count']}, success rate: {metrics['success_rate']:.1f}%")
                    
                    # Emit update to client
                    self.socketio.emit('training_update', {
                        'frame': frame,  # Assuming get_frame already returns base64
                        'metrics': metrics,
                        'reward': float(reward)
                    })

                    # Check if max episodes reached or success rate achieved
                    if metrics['training_complete']:
                        print("Training complete by metrics! Stopping training loop.")
                        self.is_training = False
                        self.socketio.emit('training_complete', {
                            'message': 'Training complete!',
                            'metrics': metrics,
                            'frame': frame
                        })
                        break
                    
                    # Control update frequency - don't update too fast
                    time.sleep(0.1)
                    
                except Exception as e:
                    print(f"Error in training loop: {e}")
                    traceback.print_exc()
                    self.is_training = False
                    self.socketio.emit('training_error', {'error': str(e)})
                    break
                
        self.is_training = True
        self.training_thread = threading.Thread(target=training_loop, daemon=True)
        self.training_thread.start()
        print(f"Training thread started and is alive: {self.training_thread.is_alive()}")
        
    def stop_training(self):
        """Stop the training loop"""
        print("Stopping training loop")
        self.is_training = False
        if self.training_thread and self.training_thread.is_alive():
            # Don't join - could block, and daemon thread will terminate anyway
            print("Training thread will terminate")