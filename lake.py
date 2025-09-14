import gymnasium as gym
import numpy as np
from collections import defaultdict
import cv2
import traceback
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

class FrozenLake:
    def __init__(self, size=8, learning_rate=0.001, discount_factor=0.99, epsilon=1.0, is_slippery=False,
                 max_episodes=500, success_rate_threshold=95):
        # Environment parameters
        self.size = size
        self.is_slippery = is_slippery
        
        # Training limits
        self.max_episodes = max_episodes
        self.success_rate_threshold = success_rate_threshold
        self.training_complete = False
        
        # Generate initial random map
        self.random_map = generate_random_map(size=self.size)
        
        # Create environment
        self.env = gym.make(
            'FrozenLake-v1',
            desc=self.random_map,
            is_slippery=self.is_slippery,
            render_mode="rgb_array"
        )
        
        # Learning parameters
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.initial_epsilon = epsilon
        self.epsilon = epsilon
        self.min_epsilon = 0.01
        self.epsilon_decay = 0.995
        
        # Initialize Q-table
        self.q_table = defaultdict(lambda: np.zeros(self.env.action_space.n))
        
        # Training metrics
        self.episode_rewards = []
        self.success_rate = 0
        self.current_episode = 0
        self.optimal_path = []
        self.best_path_length = float('inf')
        
        # Initialize state
        self.state, _ = self.env.reset()

    def update_parameters(self, learning_rate=None, discount_factor=None, epsilon=None,
                        max_episodes=None, success_rate_threshold=None):
        """Update learning parameters"""
        if learning_rate is not None:
            self.learning_rate = learning_rate
        if discount_factor is not None:
            self.discount_factor = discount_factor
        if epsilon is not None:
            self.initial_epsilon = epsilon
            self.epsilon = epsilon
        if max_episodes is not None:
            self.max_episodes = max_episodes
        if success_rate_threshold is not None:
            self.success_rate_threshold = success_rate_threshold

    def randomize_map(self, size=None):
        """Generate a new random map and reset environment"""
        if size is not None:
            self.size = size
            
        # Generate new random map
        self.random_map = generate_random_map(size=self.size)
        
        # Close existing environment
        self.env.close()
        
        # Create new environment with new map
        self.env = gym.make(
            'FrozenLake-v1',
            desc=self.random_map,
            is_slippery=self.is_slippery,
            render_mode="rgb_array"
        )
        
        # Reset all training parameters
        self.state, _ = self.env.reset()
        self.optimal_path = []
        self.q_table = defaultdict(lambda: np.zeros(self.env.action_space.n))
        self.epsilon = self.initial_epsilon
        self.episode_rewards = []
        self.current_episode = 0
        self.success_rate = 0
        self.training_complete = False
        self.best_path_length = float('inf')

    def check_training_complete(self):
        """Check if training should stop"""
        # Check if maximum episodes reached
        if self.current_episode >= self.max_episodes:
            self.training_complete = True
            return True
            
        # Check if success rate threshold is met
        if len(self.episode_rewards) >= 100:  # Need at least 100 episodes to calculate reliable success rate
            if self.success_rate >= self.success_rate_threshold:
                # Verify we have found an optimal path
                if self.optimal_path and len(self.optimal_path) < self.best_path_length:
                    self.best_path_length = len(self.optimal_path)
                self.training_complete = True
                return True
                
        return False

    def get_action(self, state):
        """Select action using epsilon-greedy policy"""
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        return int(np.argmax(self.q_table[state]))

    def update_q_value(self, state, action, reward, next_state, done):
        """Update Q-value using Q-learning"""
        best_next_action = np.argmax(self.q_table[next_state])
        
        # Modify reward structure
        if done and reward == 0:  # Fell in a hole
            reward = -1
        elif not done and reward == 0:  # Still moving
            reward = -0.01  # Small negative reward to encourage shorter paths
            
        # Q-learning update
        current_q = self.q_table[state][action]
        next_max_q = self.q_table[next_state][best_next_action]
        new_q = current_q + self.learning_rate * (
            reward + (0 if done else self.discount_factor * next_max_q) - current_q
        )
        self.q_table[state][action] = new_q

    def get_optimal_path(self):
        """Calculate optimal path using current Q-table"""
        state = 0  # Start state
        path = [state]
        done = False
        max_steps = self.size * self.size * 2  # Prevent infinite loops
        step_count = 0
        
        self.env.reset()
        
        while not done and step_count < max_steps:
            action = np.argmax(self.q_table[state])
            next_state, reward, done, _, _ = self.env.step(action)
            path.append(next_state)
            state = next_state
            step_count += 1
            
            if reward == 1:  # Reached goal
                if len(path) < self.best_path_length:
                    self.best_path_length = len(path)
                    self.optimal_path = path
                return path
                
        return None

    def generate_path_visualization(self):
        """Visualize the environment with optimal path"""
        frame = self.env.render()
        if frame is None:
            return None
            
        # Resize for better visualization
        frame = cv2.resize(frame, (400, 400), interpolation=cv2.INTER_NEAREST)
        
        if self.optimal_path:
            size = self.size
            cell_height = frame.shape[0] // size
            cell_width = frame.shape[1] // size
            
            # Draw path
            for i in range(len(self.optimal_path) - 1):
                current = self.optimal_path[i]
                next_state = self.optimal_path[i + 1]
                
                # Calculate positions
                current_y, current_x = current // size, current % size
                next_y, next_x = next_state // size, next_state % size
                
                # Calculate center points
                start_point = (
                    int((current_x + 0.5) * cell_width),
                    int((current_y + 0.5) * cell_height)
                )
                end_point = (
                    int((next_x + 0.5) * cell_width),
                    int((next_y + 0.5) * cell_height)
                )
                
                # Draw path
                cv2.line(frame, start_point, end_point, (255, 0, 0), 2)
                cv2.circle(frame, start_point, 5, (0, 255, 0), -1)
            
            # Draw final point
            if self.optimal_path:
                final = self.optimal_path[-1]
                final_y, final_x = final // size, final % size
                final_point = (
                    int((final_x + 0.5) * cell_width),
                    int((final_y + 0.5) * cell_height)
                )
                cv2.circle(frame, final_point, 5, (0, 0, 255), -1)
        
        return frame

    def train_episode(self):
        """Train for one episode using Q-learning"""
        try:
            # Check if training is already complete
            if self.training_complete:
                return 0.0
                
            state, _ = self.env.reset()
            total_reward = 0
            done = False
            
            while not done:
                # Select and take action
                action = self.get_action(state)
                next_state, reward, done, _, _ = self.env.step(action)
                
                # Update Q-value
                self.update_q_value(state, action, reward, next_state, done)
                
                # Update state and reward
                state = next_state
                total_reward += reward
            
            # Update metrics
            self.episode_rewards.append(total_reward)
            self.current_episode += 1
            
            # Calculate success rate
            recent_episodes = min(100, len(self.episode_rewards))
            successes = sum(1 for r in self.episode_rewards[-recent_episodes:] if r > 0)
            self.success_rate = (successes / recent_episodes) * 100
            
            # Decay epsilon
            self.epsilon = max(
                self.min_epsilon,
                self.epsilon * self.epsilon_decay
            )
            
            # Update optimal path if episode was successful
            if total_reward > 0:
                self.get_optimal_path()
                
            # Check if training should stop
            self.check_training_complete()
            
            return total_reward
            
        except Exception as e:
            print(f"Error in train_episode: {e}")
            traceback.print_exc()
            return 0.0

    def get_frame(self):
        """Get current frame with visualization"""
        try:
            return self.generate_path_visualization()
        except Exception as e:
            print(f"Error getting frame: {e}")
            return None

    def get_metrics(self):
        """Get current training metrics"""
        if not self.episode_rewards:
            return {
                'average_reward': 0,
                'episode_count': 0,
                'success_rate': 0,
                'epsilon': self.epsilon,
                'training_complete': self.training_complete,
                'best_path_length': self.best_path_length if self.best_path_length != float('inf') else 0
            }
            
        recent_rewards = self.episode_rewards[-100:] if self.episode_rewards else [0]
        return {
            'average_reward': np.mean(recent_rewards),
            'episode_count': self.current_episode,
            'success_rate': self.success_rate,
            'epsilon': self.epsilon,
            'training_complete': self.training_complete,
            'best_path_length': self.best_path_length if self.best_path_length != float('inf') else 0
        }

    def save_model(self):
        """Save the Q-table"""
        np.save('q_table.npy', dict(self.q_table))

    def load_model(self):
        """Load the Q-table"""
        try:
            q_table_dict = np.load('q_table.npy', allow_pickle=True).item()
            self.q_table = defaultdict(lambda: np.zeros(self.env.action_space.n), q_table_dict)
        except Exception as e:
            print(f"Error loading model: {e}")
            traceback.print_exc()