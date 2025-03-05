# Modified training_manager.py
from flask_socketio import SocketIO, emit
import threading
import time
import base64
from PIL import Image
import io
import traceback
import logging

logger = logging.getLogger(__name__)

class TrainingManager:
    def __init__(self, socketio=None):
        self.is_training = False
        self.training_thread = None
        self.socketio = socketio
        self.last_update = {}
        
    def start_training(self, agent):
        """Start training loop in a separate thread"""
        logger.info("Starting training loop")
        agent.training_complete = False  # Reset training flag
        
        def training_loop():
            logger.info("Training loop started")
            
            while self.is_training:
                try:
                    # Check if training is complete
                    if agent.training_complete:
                        logger.info("Training complete! Stopping training loop.")
                        self.is_training = False
                        
                        # Get final metrics and frame
                        final_metrics = agent.get_metrics()
                        final_frame = agent.get_frame()
                        
                        # Store last update
                        self.last_update = {
                            'message': 'Training complete!',
                            'metrics': final_metrics,
                            'frame': final_frame
                        }
                        
                        # Send completion notification if Socket.IO is available
                        if self.socketio:
                            self.socketio.emit('training_complete', self.last_update)
                        break

                    # Run training episode
                    reward = agent.train_episode()
                    metrics = agent.get_metrics()
                    frame = agent.get_frame()
                    
                    # Store update for REST API
                    self.last_update = {
                        'frame': frame,
                        'metrics': metrics,
                        'reward': float(reward)
                    }
                    
                    # Log progress
                    if metrics['episode_count'] % 10 == 0:
                        logger.info(f"Episode {metrics['episode_count']}, success rate: {metrics['success_rate']:.1f}%")
                    
                    # Emit update to client if Socket.IO is available
                    if self.socketio:
                        self.socketio.emit('training_update', self.last_update)

                    # Check if max episodes reached or success rate achieved
                    if metrics['training_complete']:
                        logger.info("Training complete by metrics! Stopping training loop.")
                        self.is_training = False
                        
                        # Update completion info
                        self.last_update = {
                            'message': 'Training complete!',
                            'metrics': metrics,
                            'frame': frame
                        }
                        
                        # Send notification if Socket.IO is available
                        if self.socketio:
                            self.socketio.emit('training_complete', self.last_update)
                        break
                    
                    # Control update frequency - don't update too fast
                    time.sleep(0.1)
                    
                except Exception as e:
                    logger.error(f"Error in training loop: {e}")
                    traceback.print_exc()
                    self.is_training = False
                    
                    error_info = {'error': str(e)}
                    self.last_update = error_info
                    
                    # Send error if Socket.IO is available
                    if self.socketio:
                        self.socketio.emit('training_error', error_info)
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
            
    def get_latest_update(self):
        """Return the latest training update for REST API access"""
        return self.last_update