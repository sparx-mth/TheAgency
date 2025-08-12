"""
train_dqn_interactive.py - Updated for new unified state/action space

Interactive DQN training script for the SLAM environment using Stable Baselines3.
This script provides an interactive interface to configure and train/test DQN agents.
"""

import os
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

# NO LONGER NEED SingleAgentWrapper
from environments.slam_env import MultiAgentSLAMEnv
from sensors.camera_sensor import CameraSensor


class InteractiveTrainer:
    """Interactive trainer for DQN SLAM agents."""

    def __init__(self):
        self.config = {}

    def get_user_input(self, prompt, default=None, input_type=str, options=None):
        """Get user input with default value and type conversion."""
        if default is not None:
            prompt += f" [{default}]"
        if options:
            prompt += f" {options}"
        prompt += ": "

        user_input = input(prompt).strip()

        if not user_input and default is not None:
            return default

        if input_type == bool:
            return user_input.lower() in ['y', 'yes', 'true', '1']
        elif input_type == int:
            try:
                return int(user_input)
            except ValueError:
                print(f"Invalid input. Using default: {default}")
                return default
        elif input_type == float:
            try:
                return float(user_input)
            except ValueError:
                print(f"Invalid input. Using default: {default}")
                return default
        return user_input

    def get_available_maps(self):
        """Get list of available map files."""
        maps = {}
        map_dirs = [
            '/home/nadavc/PycharmProjects/TheAgency_workspace/resources/planner/maps',
            'resources/planner/maps',
            'maps',
            'environments/maps',
            '../maps',
            './'
        ]

        for map_dir in map_dirs:
            if os.path.exists(map_dir):
                for file in os.listdir(map_dir):
                    if file.endswith(('.txt', '.npy', '.map', '.csv')):
                        maps[file] = os.path.join(map_dir, file)

        return maps

    def configure_training(self):
        """Interactively configure training parameters."""
        print("\n" + "="*60)
        print("TRAINING CONFIGURATION")
        print("="*60)

        # Map selection
        print("\n--- Map Configuration ---")
        map_choice = self.get_user_input(
            "Map type (1=random, 2=fixed)", "1", str
        )

        if map_choice == "2":
            # Fixed map mode
            self.config['randomize_maps'] = False

            # Show available maps
            available_maps = self.get_available_maps()
            if available_maps:
                print("\nAvailable maps:")
                map_list = list(available_maps.items())
                for i, (name, path) in enumerate(map_list, 1):
                    print(f"  {i}. {name}")
                print(f"  {len(map_list)+1}. Create a simple test map")
                print(f"  {len(map_list)+2}. Enter custom path")

                map_selection = self.get_user_input(
                    f"Select map (1-{len(map_list)+2})", "1", int
                )

                if 1 <= map_selection <= len(map_list):
                    self.config['map_path'] = map_list[map_selection-1][1]
                    print(f"Selected map: {map_list[map_selection-1][0]}")
                elif map_selection == len(map_list) + 1:
                    # Create a simple test map
                    self.config['map_path'] = None
                    self.config['use_simple_map'] = True
                    print("Will use a simple generated test map")
                else:
                    custom_path = self.get_user_input("Enter map file path", "", str)
                    if os.path.exists(custom_path):
                        self.config['map_path'] = custom_path
                    else:
                        print("File not found. Will use random maps instead.")
                        self.config['randomize_maps'] = True
            else:
                print("No map files found. Using random maps.")
                self.config['randomize_maps'] = True
        else:
            # Random map mode
            self.config['randomize_maps'] = True
            self.config['map_path'] = None

        # Environment configuration
        print("\n--- Environment Configuration ---")

        # Only ask for dimensions if using random maps or simple test map
        if self.config.get('randomize_maps', True) or self.config.get('use_simple_map', False):
            self.config['grid_width'] = self.get_user_input(
                "Grid width", 20, int
            )
            self.config['grid_height'] = self.get_user_input(
                "Grid height", 20, int
            )
        else:
            # Using a fixed map - dimensions will be determined by the map file
            if self.config.get('map_path'):
                try:
                    # Try to load the map to get its dimensions
                    import numpy as np
                    map_path = self.config['map_path']
                    if map_path.endswith('.npy'):
                        map_data = np.load(map_path)
                    elif map_path.endswith('.csv'):
                        map_data = np.loadtxt(map_path, delimiter=',', dtype=np.int8)
                    else:
                        map_data = np.loadtxt(map_path, dtype=np.int8)

                    height, width = map_data.shape
                    self.config['grid_width'] = width
                    self.config['grid_height'] = height
                    print(f"Map dimensions: {width}x{height} (from loaded map)")
                except Exception as e:
                    print(f"Could not read map dimensions: {e}")
                    print("Using default dimensions (will be overridden by map file)")
                    self.config['grid_width'] = 20
                    self.config['grid_height'] = 20
            else:
                # Fallback to defaults
                self.config['grid_width'] = 20
                self.config['grid_height'] = 20

        self.config['max_steps'] = self.get_user_input(
            "Max steps per episode", 500, int
        )

        # Sensor configuration
        print("\n--- Sensor Configuration ---")
        self.config['sensor_range'] = self.get_user_input(
            "Sensor max range", 8, int
        )
        self.config['sensor_fov'] = self.get_user_input(
            "Sensor field of view (degrees)", 60, int
        )
        self.config['sensor_rays'] = self.get_user_input(
            "Number of sensor rays", 20, int
        )

        # Reward configuration
        print("\n--- Reward Configuration ---")
        self.config['discovery_reward'] = self.get_user_input(
            "Discovery reward per cell", 0.1, float
        )
        self.config['collision_penalty'] = self.get_user_input(
            "Collision penalty", -0.5, float
        )
        self.config['step_penalty'] = self.get_user_input(
            "Step penalty", -0.001, float
        )
        self.config['completion_bonus'] = self.get_user_input(
            "Completion bonus", 10.0, float
        )

        # Training configuration
        print("\n--- Training Configuration ---")
        self.config['total_timesteps'] = self.get_user_input(
            "Total training timesteps", 100000, int
        )
        self.config['learning_rate'] = self.get_user_input(
            "Learning rate", 1e-4, float
        )
        self.config['buffer_size'] = self.get_user_input(
            "Replay buffer size", 50000, int
        )
        self.config['batch_size'] = self.get_user_input(
            "Batch size", 32, int
        )
        self.config['exploration_fraction'] = self.get_user_input(
            "Exploration fraction (0-1)", 0.1, float
        )

        # Parallel environments
        self.config['n_envs'] = self.get_user_input(
            "Number of parallel environments for training", 4, int
        )

        # Evaluation
        self.config['eval_freq'] = self.get_user_input(
            "Evaluation frequency (steps)", 5000, int
        )
        self.config['n_eval_episodes'] = self.get_user_input(
            "Number of evaluation episodes", 10, int
        )

        # Rendering during training
        self.config['render_training'] = self.get_user_input(
            "Show visualization during training? (y/n)", False, bool
        )

        if self.config['render_training']:
            print("  Note: Visualization will slow down training and disable parallel environments")

        # Save configuration
        self.config['save_path'] = self.get_user_input(
            "Model save path", "models/interactive", str
        )

    def configure_testing(self):
        """Interactively configure testing parameters."""
        print("\n" + "="*60)
        print("TESTING CONFIGURATION")
        print("="*60)

        # Model selection
        print("\nAvailable models:")
        model_files = []
        if os.path.exists("models"):
            for root, dirs, files in os.walk("models"):
                for file in files:
                    if file.endswith('.zip'):
                        model_path = os.path.join(root, file[:-4])
                        model_files.append(model_path)
                        print(f"  - {model_path}")

        if not model_files:
            print("  No models found. Please train a model first.")
            return False

        self.config['model_path'] = self.get_user_input(
            "\nModel path", "models/best/best_model", str
        )

        # Check if model exists
        model_file = self.config['model_path'] if self.config['model_path'].endswith('.zip') else f"{self.config['model_path']}.zip"
        if not os.path.exists(model_file):
            print(f"Error: Model not found at {model_file}")
            return False

        # Test configuration
        self.config['test_episodes'] = self.get_user_input(
            "Number of test episodes", 5, int
        )
        self.config['render_test'] = self.get_user_input(
            "Show visualization? (y/n)", True, bool
        )

        # Environment configuration for testing
        print("\n--- Test Environment Configuration ---")

        # Map selection for testing
        map_choice = self.get_user_input(
            "Map type (1=random, 2=fixed, 3=same as training)", "3", str
        )

        if map_choice == "2":
            # Fixed map mode
            self.config['randomize_maps'] = False
            available_maps = self.get_available_maps()

            if available_maps:
                print("\nAvailable maps:")
                map_list = list(available_maps.items())
                for i, (name, path) in enumerate(map_list, 1):
                    print(f"  {i}. {name}")

                map_selection = self.get_user_input(
                    f"Select map (1-{len(map_list)})", "1", int
                )

                if 1 <= map_selection <= len(map_list):
                    self.config['map_path'] = map_list[map_selection-1][1]
                    print(f"Selected map: {map_list[map_selection-1][0]}")
            else:
                print("No maps found. Using random maps.")
                self.config['randomize_maps'] = True
                self.config['grid_width'] = self.get_user_input("Grid width", 20, int)
                self.config['grid_height'] = self.get_user_input("Grid height", 20, int)
        elif map_choice == "1":
            self.config['randomize_maps'] = True
            self.config['map_path'] = None
            self.config['grid_width'] = self.get_user_input("Grid width", 20, int)
            self.config['grid_height'] = self.get_user_input("Grid height", 20, int)
        else:
            # Use same configuration as training (if available)
            self.config['randomize_maps'] = True
            self.config['map_path'] = None
            self.config['grid_width'] = self.get_user_input("Grid width", 20, int)
            self.config['grid_height'] = self.get_user_input("Grid height", 20, int)

        self.config['max_steps'] = self.get_user_input(
            "Max steps per episode", 500, int
        )

        return True

    def make_env(self, render=False):
        """Create environment based on configuration."""
        sensor = CameraSensor(
            max_range=self.config.get('sensor_range', 8),
            fov_deg=self.config.get('sensor_fov', 60),
            num_rays=self.config.get('sensor_rays', 20)
        )

        # Handle map configuration
        map_path = self.config.get('map_path', None)
        randomize = self.config.get('randomize_maps', True)

        # If using a simple test map and no path specified
        if self.config.get('use_simple_map', False) and not map_path:
            # Environment will create a simple default map when randomize=False and no path
            map_path = None
            randomize = False

        # Use MultiAgentSLAMEnv with num_agents=1 for single agent
        env = MultiAgentSLAMEnv(
            width=self.config.get('grid_width', 20),
            height=self.config.get('grid_height', 20),
            num_agents=1,  # Single agent for DQN training
            max_steps=self.config.get('max_steps', 500),
            map_path=map_path,
            randomize=randomize,
            render_mode='human' if render else None,
            sensor_config={0: sensor},
            discovery_reward=self.config.get('discovery_reward', 0.1),
            collision_penalty=self.config.get('collision_penalty', -0.5),
            step_penalty=self.config.get('step_penalty', -0.001),
            completion_bonus=self.config.get('completion_bonus', 10.0),
        )
        
        return env
    
    def train(self):
        """Train the DQN model with configured parameters."""
        print("\n" + "="*60)
        print("STARTING TRAINING")
        print("="*60)
        
        # Create directories
        os.makedirs(self.config['save_path'], exist_ok=True)
        os.makedirs(f"{self.config['save_path']}/checkpoints", exist_ok=True)
        os.makedirs("logs", exist_ok=True)
        
        # Create training environments
        print("\nCreating training environments...")
        n_envs = self.config.get('n_envs', 4)
        
        if n_envs > 1 and not self.config.get('render_training', False):
            # Use multiple parallel environments for faster training
            def make_env_fn():
                return Monitor(self.make_env(render=False))
            train_envs = SubprocVecEnv([make_env_fn for _ in range(n_envs)])
            print(f" Created {n_envs} parallel training environments")
        else:
            # Single environment (required if rendering)
            train_env = Monitor(self.make_env(render=self.config.get('render_training', False)))
            train_envs = DummyVecEnv([lambda: train_env])
            print(" Created single training environment")
        
        # Create evaluation environment
        print("Creating evaluation environment...")
        eval_env = Monitor(self.make_env(render=False))
        eval_env = DummyVecEnv([lambda: eval_env])
        print(" Created evaluation environment")
        
        # Create DQN model
        print("\nInitializing DQN model...")
        
        # Check if tensorboard is available
        try:
            import tensorboard
            tb_log = "./logs/tensorboard/"
            print(" TensorBoard logging enabled")
        except ImportError:
            tb_log = None
            print(" TensorBoard not installed - logging disabled")
            print("  Install with: pip install tensorboard")
        
        model = DQN(
            policy="MultiInputPolicy",  # Required for dict observation spaces
            env=train_envs,
            learning_rate=self.config.get('learning_rate', 1e-4),
            buffer_size=self.config.get('buffer_size', 50000),
            learning_starts=1000,
            batch_size=self.config.get('batch_size', 32),
            tau=1.0,
            gamma=0.99,
            train_freq=4,
            gradient_steps=1,
            target_update_interval=1000,
            exploration_fraction=self.config.get('exploration_fraction', 0.1),
            exploration_initial_eps=1.0,
            exploration_final_eps=0.05,
            verbose=1,
            tensorboard_log=tb_log,
        )
        print(" DQN model initialized")
        
        # Setup callbacks
        checkpoint_callback = CheckpointCallback(
            save_freq=10000,
            save_path=f"{self.config['save_path']}/checkpoints/",
            name_prefix="dqn_slam"
        )
        
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=f"{self.config['save_path']}/best/",
            log_path="./logs/eval/",
            eval_freq=self.config.get('eval_freq', 5000),
            n_eval_episodes=self.config.get('n_eval_episodes', 10),
            deterministic=True,
            render=False
        )
        
        # Start training
        print("\n" + "-"*60)
        print("Training in progress...")
        print(f"Total timesteps: {self.config['total_timesteps']}")
        if tb_log:
            print(f"You can monitor progress with: tensorboard --logdir ./logs/tensorboard/")
        print("-"*60 + "\n")
        
        # Check if progress bar dependencies are available
        try:
            import tqdm
            import rich
            use_progress_bar = True
        except ImportError:
            use_progress_bar = False
            print(" Progress bar disabled - install with: pip install tqdm rich")
            print("")
        
        try:
            model.learn(
                total_timesteps=self.config['total_timesteps'],
                callback=[checkpoint_callback, eval_callback],
                progress_bar=use_progress_bar
            )
            
            # Save final model
            final_path = f"{self.config['save_path']}/final_model"
            model.save(final_path)
            
            print("\n" + "="*60)
            print("TRAINING COMPLETE!")
            print("="*60)
            print(f" Best model saved at: {self.config['save_path']}/best/best_model.zip")
            print(f" Final model saved at: {final_path}.zip")
            
        except KeyboardInterrupt:
            print("\n\nTraining interrupted by user")
            interrupt_path = f"{self.config['save_path']}/interrupted_model"
            model.save(interrupt_path)
            print(f" Model saved at: {interrupt_path}.zip")
        
        finally:
            train_envs.close()
            eval_env.close()
    
    def test(self):
        """Test a trained model."""
        print("\n" + "="*60)
        print("STARTING TESTING")
        print("="*60)
        
        # Load model
        print(f"\nLoading model from {self.config['model_path']}...")
        model = DQN.load(self.config['model_path'])
        print(" Model loaded successfully")
        
        # Create test environment
        env = self.make_env(render=self.config.get('render_test', True))
        
        # Run test episodes
        total_rewards = []
        progress_scores = []
        collision_counts = []
        
        for episode in range(self.config['test_episodes']):
            print(f"\n--- Episode {episode + 1}/{self.config['test_episodes']} ---")
            
            obs, info = env.reset()
            episode_reward = 0
            done = False
            step = 0
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                episode_reward += reward
                step += 1
                
                if self.config.get('render_test', True):
                    env.render()
                
                # Print progress periodically
                if step % 100 == 0:
                    print(f"  Step {step}: Progress {info['progress']*100:.1f}%")
            
            # Episode statistics
            total_rewards.append(episode_reward)
            progress_scores.append(info['progress'])
            collision_counts.append(info['collision_counts'][0])
            
            print(f"  Episode finished:")
            print(f"    - Total reward: {episode_reward:.2f}")
            print(f"    - Final progress: {info['progress']*100:.1f}%")
            print(f"    - Steps taken: {step}")
            print(f"    - Collisions: {info['collision_counts'][0]}")
        
        # Summary statistics
        print("\n" + "="*60)
        print("TESTING COMPLETE - SUMMARY")
        print("="*60)
        print(f"Average reward:    {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
        print(f"Average progress:  {np.mean(progress_scores)*100:.1f}% ± {np.std(progress_scores)*100:.1f}%")
        print(f"Average collisions: {np.mean(collision_counts):.1f} ± {np.std(collision_counts):.1f}")
        print(f"Best episode:      {np.max(total_rewards):.2f} reward, {np.max(progress_scores)*100:.1f}% progress")
        
        env.close()
    
    def run(self):
        """Main interactive interface."""
        print("\n" + "="*60)
        print("INTERACTIVE DQN SLAM TRAINER")
        print("="*60)
        
        # Main mode selection
        print("\nWhat would you like to do?")
        print("1. Train a new model")
        print("2. Test an existing model")
        print("3. Quick demo (minimal training + test)")
        print("4. Exit")
        
        choice = self.get_user_input("Select option (1-4)", "1", str)
        
        if choice == "1":
            # Training mode
            use_defaults = self.get_user_input(
                "\nUse default configuration? (y/n)", True, bool
            )
            
            if use_defaults:
                print("\nUsing default configuration...")
                # Set reasonable defaults
                self.config = {
                    'grid_width': 20,
                    'grid_height': 20,
                    'max_steps': 500,
                    'randomize_maps': True,
                    'sensor_range': 8,
                    'sensor_fov': 60,
                    'sensor_rays': 20,
                    'discovery_reward': 0.1,
                    'collision_penalty': -0.5,
                    'step_penalty': -0.001,
                    'completion_bonus': 10.0,
                    'total_timesteps': 100000,
                    'learning_rate': 1e-4,
                    'buffer_size': 50000,
                    'batch_size': 32,
                    'exploration_fraction': 0.1,
                    'n_envs': 4,
                    'eval_freq': 5000,
                    'n_eval_episodes': 10,
                    'render_training': False,
                    'save_path': 'models/interactive'
                }
            else:
                self.configure_training()
            
            # Confirm and start training
            print("\n" + "-"*60)
            print("Configuration Summary:")
            print(f"  Environment: {self.config['grid_width']}x{self.config['grid_height']} grid")
            print(f"  Training steps: {self.config['total_timesteps']}")
            print(f"  Parallel environments: {self.config.get('n_envs', 1)}")
            print(f"  Save path: {self.config['save_path']}")
            print("-"*60)
            
            proceed = self.get_user_input("\nProceed with training? (y/n)", True, bool)
            if proceed:
                self.train()
            
        elif choice == "2":
            # Testing mode
            if self.configure_testing():
                self.test()
            
        elif choice == "3":
            # Quick demo mode
            print("\nRunning quick demo (5000 steps training)...")
            self.config = {
                'grid_width': 10,
                'grid_height': 10,
                'max_steps': 200,
                'randomize_maps': True,
                'sensor_range': 5,
                'sensor_fov': 60,
                'sensor_rays': 10,
                'discovery_reward': 0.1,
                'collision_penalty': -0.5,
                'step_penalty': -0.001,
                'completion_bonus': 5.0,
                'total_timesteps': 5000,
                'learning_rate': 1e-3,
                'buffer_size': 5000,
                'batch_size': 32,
                'exploration_fraction': 0.3,
                'n_envs': 1,
                'eval_freq': 1000,
                'n_eval_episodes': 3,
                'render_training': False,
                'save_path': 'models/demo'
            }
            
            # Train
            self.train()
            
            # Test
            self.config['model_path'] = 'models/demo/best/best_model'
            self.config['test_episodes'] = 3
            self.config['render_test'] = True
            
            print("\n\nNow testing the trained model...")
            input("Press Enter to continue...")
            self.test()
            
        elif choice == "4":
            print("\nExiting...")
            return
        else:
            print("\nInvalid option. Exiting...")


if __name__ == "__main__":
    trainer = InteractiveTrainer()
    trainer.run()