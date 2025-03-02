from flask_socketio import SocketIO, emit
import threading
import time
import base64
from PIL import Image
import io
import traceback
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrainingManager:
    def __init__(self, socketio):
        self.is_training = False
        self.training_thread = None
        self.socketio = socketio
        logger.info("Training manager initialized")
        
    def start_training(self, agent):
        """Start training loop in a separate thread"""
        logger.info("Starting training loop")
        agent.training_complete = False  # Reset training flag
        
        def training_loop():
            logger.info("Training loop started")
            
            # Local reference to flag for thread safety
            local_is_training = True
            
            while local_is_training and self.is_training:
                try:
                    # Check if training is complete
                    if agent.training_complete:
                        logger.info("Training complete! Stopping training loop.")
                        self.is_training = False
                        local_is_training = False
                        
                        # Get final metrics and frame
                        final_metrics = agent.get_metrics()
                        final_frame = agent.get_frame()
                        
                        # Send completion notification - with error handling
                        try:
                            if self.socketio:
                                self.socketio.emit('training_complete', {
                                    'message': 'Training complete!',
                                    'metrics': final_metrics,
                                    'frame': final_frame
                                })
                        except Exception as emit_error:
                            logger.error(f"Error emitting completion: {emit_error}")
                        break

                    # Run training episode
                    reward = agent.train_episode()
                    metrics = agent.get_metrics()
                    frame = agent.get_frame()
                    
                    # Log progress
                    if metrics['episode_count'] % 10 == 0:
                        logger.info(f"Episode {metrics['episode_count']}, success rate: {metrics['success_rate']:.1f}%")
                    
                    # Emit update to client - with error handling
                    try:
                        if self.socketio:
                            self.socketio.emit('training_update', {
                                'frame': frame,
                                'metrics': metrics,
                                'reward': float(reward)
                            })
                    except Exception as emit_error:
                        logger.error(f"Error emitting update: {emit_error}")

                    # Check if max episodes reached or success rate achieved
                    if metrics['training_complete']:
                        logger.info("Training complete by metrics! Stopping training loop.")
                        self.is_training = False
                        local_is_training = False
                        
                        try:
                            if self.socketio:
                                self.socketio.emit('training_complete', {
                                    'message': 'Training complete!',
                                    'metrics': metrics,
                                    'frame': frame
                                })
                        except Exception as emit_error:
                            logger.error(f"Error emitting completion: {emit_error}")
                        break
                    
                    # Control update frequency - use more conservative timing for Cloud Run
                    time.sleep(0.25)
                    
                except Exception as e:
                    logger.error(f"Error in training loop: {e}")
                    traceback.print_exc()
                    # Don't break the loop, try to continue
                    time.sleep(1.0)  # Wait longer after an error
                
        # Set flags and start thread with daemon=True for cloud environment
        self.is_training = True
        self.training_thread = threading.Thread(target=training_loop, daemon=True)
        self.training_thread.start()
        logger.info(f"Training thread started and is alive: {self.training_thread.is_alive()}")
        
    def stop_training(self):
        """Stop the training loop"""
        logger.info("Stopping training loop")
        self.is_training = False
        # No need to join the thread in Cloud Run environment