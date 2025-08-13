"""
train_ppo_interactive.py - PPO training for MultiDiscrete action space

Interactive PPO training script for the SLAM environment using Stable Baselines3.
PPO supports MultiDiscrete action spaces, making it ideal for multi-agent environments.
"""

import os
import sys
import numpy as np

# Suppress all gym-related warnings before importing anything else
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*Gym.*")
warnings.filterwarnings("ignore", message=".*gym.*")
warnings.filterwarnings("ignore", category=UserWarning, module="gym")

# Redirect stderr temporarily during imports to suppress remaining warnings
import io
old_stderr = sys.stderr
sys.stderr = io.StringIO()

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env

# Restore stderr
sys.stderr = old_stderr

from environments.slam_env import MultiAgentSLAMEnv
from sensors.camera_sensor import CameraSensor
from sensors.lidar_sensor import LidarSensor


class InteractivePPOTrainer:
    """Interactive trainer for PPO SLAM agents with MultiDiscrete action support."""

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
            '/home/user/nadav/TheAgency/resources/planner/maps',
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
        print("PPO TRAINING CONFIGURATION")
        print("="*60)

        # Multi-agent configuration
        print("\n--- Multi-Agent Configuration ---")
        self.config['num_agents'] = self.get_user_input(
            "Number of agents (1-5)", 1, int
        )

        # Ask about heterogeneous sensors only if multiple agents
        if self.config['num_agents'] > 1:
            self.config['heterogeneous_sensors'] = self.get_user_input(
                "Use different sensor types for agents? (y/n)", False, bool
            )
        else:
            self.config['heterogeneous_sensors'] = False

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
                "Grid width", 25, int
            )
            self.config['grid_height'] = self.get_user_input(
                "Grid height", 25, int
            )
        else:
            # Try to get dimensions from map file
            if self.config.get('map_path'):
                try:
                    import numpy as np
                    map_path = self.config['map_path']
                    if map_path.endswith('.npy'):
                        map_data = np.load(map_path)
                    else:
                        map_data = np.loadtxt(map_path, dtype=np.int8)

                    height, width = map_data.shape
                    self.config['grid_width'] = width
                    self.config['grid_height'] = height
                    print(f"Map dimensions: {width}x{height} (from loaded map)")
                except Exception as e:
                    print(f"Could not read map dimensions: {e}")
                    self.config['grid_width'] = 25
                    self.config['grid_height'] = 25

        self.config['max_steps'] = self.get_user_input(
            "Max steps per episode", 1000, int
        )

        # Sensor configuration
        print("\n--- Sensor Configuration ---")
        if not self.config.get('heterogeneous_sensors', False):
            # All agents use same sensor type
            sensor_type = self.get_user_input(
                "Sensor type (1=Camera, 2=Lidar)", "1", str
            )

            if sensor_type == "1":
                self.config['sensor_type'] = 'camera'
                self.config['sensor_range'] = self.get_user_input(
                    "Camera max range", 3, int
                )
                self.config['sensor_fov'] = self.get_user_input(
                    "Camera field of view (degrees)", 30, int
                )
                self.config['sensor_rays'] = self.get_user_input(
                    "Number of camera rays", 30, int
                )
            else:
                self.config['sensor_type'] = 'lidar'
                self.config['sensor_range'] = self.get_user_input(
                    "Lidar max range", 15, int
                )
                self.config['sensor_rays'] = self.get_user_input(
                    "Number of lidar rays", 180, int
                )
        else:
            # Configure each agent's sensor
            self.config['agent_sensors'] = []
            for i in range(self.config['num_agents']):
                print(f"\nAgent {i} sensor configuration:")
                sensor_type = self.get_user_input(
                    f"Sensor type for Agent {i} (1=Camera, 2=Lidar)", "1" if i % 2 == 0 else "2", str
                )

                if sensor_type == "1":
                    sensor_config = {
                        'type': 'camera',
                        'range': self.get_user_input(f"  Camera range", 10, int),
                        'fov': self.get_user_input(f"  Camera FOV (degrees)", 60, int),
                        'rays': self.get_user_input(f"  Camera rays", 30, int)
                    }
                else:
                    sensor_config = {
                        'type': 'lidar',
                        'range': self.get_user_input(f"  Lidar range", 15, int),
                        'rays': self.get_user_input(f"  Lidar rays", 180, int)
                    }
                self.config['agent_sensors'].append(sensor_config)

        # Reward configuration
        print("\n--- Reward Configuration ---")
        self.config['discovery_reward'] = self.get_user_input(
            "Discovery reward per cell", 0.1, float
        )
        self.config['collision_penalty'] = self.get_user_input(
            "Collision penalty", 0.0, float
        )
        self.config['step_penalty'] = self.get_user_input(
            "Step penalty", 0.0, float
        )
        self.config['completion_bonus'] = self.get_user_input(
            "Completion bonus", 10.0, float
        )

        # PPO-specific hyperparameters
        print("\n--- PPO Algorithm Configuration ---")
        self.config['total_timesteps'] = self.get_user_input(
            "Total training timesteps", 2000000, int
        )
        self.config['learning_rate'] = self.get_user_input(
            "Learning rate", 3e-4, float
        )
        self.config['n_steps'] = self.get_user_input(
            "Steps per update (n_steps)", 2048, int
        )
        self.config['batch_size'] = self.get_user_input(
            "Batch size", 64, int
        )
        self.config['n_epochs'] = self.get_user_input(
            "PPO epochs per update", 10, int
        )
        self.config['gamma'] = self.get_user_input(
            "Discount factor (gamma)", 0.99, float
        )
        self.config['gae_lambda'] = self.get_user_input(
            "GAE lambda", 0.95, float
        )
        self.config['clip_range'] = self.get_user_input(
            "Clip range", 0.2, float
        )
        self.config['ent_coef'] = self.get_user_input(
            "Entropy coefficient", 0.01, float
        )

        # Parallel environments
        self.config['n_envs'] = self.get_user_input(
            "Number of parallel environments for training", 4, int
        )

        # Evaluation
        self.config['eval_freq'] = self.get_user_input(
            "Evaluation frequency (steps)", 10000, int
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
            "Model save path", "models/ppo_interactive", str
        )

    def configure_testing(self):
        """Interactively configure testing parameters."""
        print("\n" + "="*60)
        print("PPO TESTING CONFIGURATION")
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
            "\nModel path", "models/ppo_interactive/best/best_model", str
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

        # Number of agents for testing
        self.config['num_agents'] = self.get_user_input(
            "Number of agents for testing", 2, int
        )

        # Environment configuration for testing
        print("\n--- Test Environment Configuration ---")

        map_choice = self.get_user_input(
            "Map type (1=random, 2=fixed, 3=same as training)", "3", str
        )

        if map_choice == "2":
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
                self.config['grid_width'] = self.get_user_input("Grid width", 25, int)
                self.config['grid_height'] = self.get_user_input("Grid height", 25, int)
        elif map_choice == "1":
            self.config['randomize_maps'] = True
            self.config['map_path'] = None
            self.config['grid_width'] = self.get_user_input("Grid width", 25, int)
            self.config['grid_height'] = self.get_user_input("Grid height", 25, int)
        else:
            # Use same configuration as training
            self.config['randomize_maps'] = True
            self.config['map_path'] = None
            self.config['grid_width'] = self.get_user_input("Grid width", 25, int)
            self.config['grid_height'] = self.get_user_input("Grid height", 25, int)

        self.config['max_steps'] = self.get_user_input(
            "Max steps per episode", 1000, int
        )

        return True

    def create_sensor_config(self):
        """Create sensor configuration dictionary for the environment."""
        sensor_config = {}

        if self.config.get('heterogeneous_sensors', False) and 'agent_sensors' in self.config:
            # Heterogeneous sensors
            for i, sensor_cfg in enumerate(self.config['agent_sensors']):
                if sensor_cfg['type'] == 'camera':
                    sensor_config[i] = CameraSensor(
                        max_range=sensor_cfg['range'],
                        fov_deg=sensor_cfg['fov'],
                        num_rays=sensor_cfg['rays']
                    )
                else:  # lidar
                    try:
                        sensor_config[i] = LidarSensor(
                            max_range=sensor_cfg['range'],
                            num_rays=sensor_cfg['rays']
                        )
                    except:
                        # Fallback to camera if LidarSensor not available
                        print(f"LidarSensor not available for agent {i}, using Camera instead")
                        sensor_config[i] = CameraSensor(
                            max_range=sensor_cfg['range'],
                            fov_deg=360,  # Full coverage like lidar
                            num_rays=sensor_cfg['rays']
                        )
        else:
            # Homogeneous sensors
            if self.config.get('sensor_type', 'camera') == 'camera':
                default_sensor_params = {
                    'max_range': self.config.get('sensor_range', 10),
                    'fov_deg': self.config.get('sensor_fov', 60),
                    'num_rays': self.config.get('sensor_rays', 30)
                }
                # Let environment create default sensors
                sensor_config = None
            else:  # lidar
                try:
                    for i in range(self.config.get('num_agents', 1)):
                        sensor_config[i] = LidarSensor(
                            max_range=self.config.get('sensor_range', 15),
                            num_rays=self.config.get('sensor_rays', 180)
                        )
                except:
                    # Fallback to camera if LidarSensor not available
                    print("LidarSensor not available, using Camera with 360 FOV instead")
                    for i in range(self.config.get('num_agents', 1)):
                        sensor_config[i] = CameraSensor(
                            max_range=self.config.get('sensor_range', 15),
                            fov_deg=360,
                            num_rays=self.config.get('sensor_rays', 180)
                        )

        return sensor_config

    def make_env(self, render=False):
        """Create environment based on configuration."""
        sensor_config = self.create_sensor_config()

        # Default sensor params for homogeneous camera setup
        default_sensor_params = None
        if self.config.get('sensor_type', 'camera') == 'camera' and not self.config.get('heterogeneous_sensors', False):
            default_sensor_params = {
                'max_range': self.config.get('sensor_range', 10),
                'fov_deg': self.config.get('sensor_fov', 60),
                'num_rays': self.config.get('sensor_rays', 30)
            }

        # Handle map configuration
        map_path = self.config.get('map_path', None)
        randomize = self.config.get('randomize_maps', True)

        if self.config.get('use_simple_map', False) and not map_path:
            map_path = None
            randomize = False

        env = MultiAgentSLAMEnv(
            width=self.config.get('grid_width', 25),
            height=self.config.get('grid_height', 25),
            num_agents=self.config.get('num_agents', 2),
            max_steps=self.config.get('max_steps', 1000),
            map_path=map_path,
            randomize=randomize,
            render_mode='human' if render else None,
            sensor_config=sensor_config,
            default_sensor_params=default_sensor_params,
            discovery_reward=self.config.get('discovery_reward', 0.1),
            collision_penalty=self.config.get('collision_penalty', -1.0),
            step_penalty=self.config.get('step_penalty', -0.001),
            completion_bonus=self.config.get('completion_bonus', 10.0),
        )

        return env

    def train(self):
        """Train the PPO model with configured parameters."""
        print("\n" + "="*60)
        print("STARTING PPO TRAINING")
        print("="*60)

        # Create directories
        os.makedirs(self.config['save_path'], exist_ok=True)
        os.makedirs(f"{self.config['save_path']}/checkpoints", exist_ok=True)
        os.makedirs("logs", exist_ok=True)

        # Create training environments
        print("\nCreating training environments...")
        n_envs = self.config.get('n_envs', 4)

        # Temporarily suppress warnings during environment creation
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            if n_envs > 1 and not self.config.get('render_training', False):
                # Use multiple parallel environments
                def make_env_fn():
                    return Monitor(self.make_env(render=False))
                train_envs = SubprocVecEnv([make_env_fn for _ in range(n_envs)])
                print(f" Created {n_envs} parallel training environments")
                print(f" Total agents training: {n_envs * self.config.get('num_agents', 2)}")
            else:
                # Single environment
                train_env = Monitor(self.make_env(render=self.config.get('render_training', False)))
                train_envs = DummyVecEnv([lambda: train_env])
                print(f" Created single training environment with {self.config.get('num_agents', 2)} agents")

            # Create evaluation environment
            print("Creating evaluation environment...")
            eval_env = Monitor(self.make_env(render=False))
            eval_env = DummyVecEnv([lambda: eval_env])
            print(" Created evaluation environment")

        # Create PPO model
        print("\nInitializing PPO model...")

        # Check if tensorboard is available
        try:
            import tensorboard
            tb_log = "./logs/tensorboard/"
            print(" TensorBoard logging enabled")
        except ImportError:
            tb_log = None
            print(" TensorBoard not installed - logging disabled")
            print("  Install with: pip install tensorboard")

        model = PPO(
            policy="MultiInputPolicy",  # For dict observation spaces
            env=train_envs,
            learning_rate=self.config.get('learning_rate', 3e-4),
            n_steps=self.config.get('n_steps', 2048),
            batch_size=self.config.get('batch_size', 64),
            n_epochs=self.config.get('n_epochs', 10),
            gamma=self.config.get('gamma', 0.99),
            gae_lambda=self.config.get('gae_lambda', 0.95),
            clip_range=self.config.get('clip_range', 0.2),
            ent_coef=self.config.get('ent_coef', 0.01),
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=1,
            tensorboard_log=tb_log,
        )
        print(" PPO model initialized")
        print(f" Action space: {train_envs.action_space}")
        print(f" Observation space: {train_envs.observation_space}")

        # Setup callbacks
        checkpoint_callback = CheckpointCallback(
            save_freq=10000,
            save_path=f"{self.config['save_path']}/checkpoints/",
            name_prefix="ppo_slam"
        )

        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=f"{self.config['save_path']}/best/",
            log_path="./logs/eval/",
            eval_freq=self.config.get('eval_freq', 10000),
            n_eval_episodes=self.config.get('n_eval_episodes', 10),
            deterministic=True,
            render=False
        )

        # Start training
        print("\n" + "-"*60)
        print("Training in progress...")
        print(f"Total timesteps: {self.config['total_timesteps']}")
        print(f"Number of agents per environment: {self.config.get('num_agents', 2)}")
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
        print("STARTING PPO TESTING")
        print("="*60)

        # Load model
        print(f"\nLoading model from {self.config['model_path']}...")
        model = PPO.load(self.config['model_path'])
        print(" Model loaded successfully")

        # Create test environment
        env = self.make_env(render=self.config.get('render_test', True))

        # Run test episodes
        total_rewards = []
        progress_scores = []
        collision_counts_all = []

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
            collision_counts_all.append(sum(info['collision_counts']))

            print(f"  Episode finished:")
            print(f"    - Total reward: {episode_reward:.2f}")
            print(f"    - Final progress: {info['progress']*100:.1f}%")
            print(f"    - Steps taken: {step}")
            print(f"    - Total collisions: {sum(info['collision_counts'])}")

            # Per-agent statistics
            if self.config.get('num_agents', 1) > 1:
                print(f"    - Per-agent collisions: {info['collision_counts']}")

        # Summary statistics
        print("\n" + "="*60)
        print("TESTING COMPLETE - SUMMARY")
        print("="*60)
        print(f"Average reward:     {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
        print(f"Average progress:   {np.mean(progress_scores)*100:.1f}% ± {np.std(progress_scores)*100:.1f}%")
        print(f"Average collisions: {np.mean(collision_counts_all):.1f} ± {np.std(collision_counts_all):.1f}")
        print(f"Best episode:       {np.max(total_rewards):.2f} reward, {np.max(progress_scores)*100:.1f}% progress")

        env.close()

    def run(self):
        """Main interactive interface."""
        print("\n" + "="*60)
        print("INTERACTIVE PPO SLAM TRAINER")
        print("Multi-Agent Support with MultiDiscrete Actions")
        print("="*60)

        # Main mode selection
        print("\nWhat would you like to do?")
        print("1. Train a new model")
        print("2. Test an existing model")
        print("3. Quick demo (minimal training + test)")
        print("4. Multi-agent demo (3 agents with heterogeneous sensors)")
        print("5. Exit")

        choice = self.get_user_input("Select option (1-5)", "1", str)

        if choice == "1":
            # Training mode
            use_defaults = self.get_user_input(
                "\nUse default configuration? (y/n)", True, bool
            )

            if use_defaults:
                print("\nUsing default configuration...")
                # Set defaults based on your preferences
                self.config = {
                    'num_agents': 1,
                    'heterogeneous_sensors': False,
                    'grid_width': 10,
                    'grid_height': 10,
                    'max_steps': 1000,
                    'randomize_maps': False,
                    'map_path': None,  # Will need to be set based on available maps
                    'use_simple_map': False,
                    'sensor_type': 'camera',
                    'sensor_range': 3,
                    'sensor_fov': 30,
                    'sensor_rays': 30,
                    'discovery_reward': 0.1,
                    'collision_penalty': 0.0,
                    'step_penalty': 0.0,
                    'completion_bonus': 10.0,
                    'total_timesteps': 2000000,
                    'learning_rate': 3e-4,
                    'n_steps': 2048,
                    'batch_size': 64,
                    'n_epochs': 10,
                    'gamma': 0.99,
                    'gae_lambda': 0.95,
                    'clip_range': 0.2,
                    'ent_coef': 0.01,
                    'n_envs': 4,
                    'eval_freq': 10000,
                    'n_eval_episodes': 10,
                    'render_training': False,
                    'save_path': 'models/ppo_interactive'
                }

                # Try to find house_map_10.txt automatically
                available_maps = self.get_available_maps()
                for name, path in available_maps.items():
                    if 'house_map_10' in name:
                        self.config['map_path'] = path
                        print(f"  Using map: {name}")
                        break

                if not self.config['map_path']:
                    print("  Note: house_map_10.txt not found, will use random maps")
                    self.config['randomize_maps'] = True
            else:
                self.configure_training()

            # Confirm and start training
            print("\n" + "-"*60)
            print("Configuration Summary:")
            print("-"*60)
            print(f"  Agents: {self.config.get('num_agents', 2)}")
            print(f"  Environment: {self.config['grid_width']}x{self.config['grid_height']} grid")
            print(f"  Map: {'Random' if self.config.get('randomize_maps', True) else self.config.get('map_path', 'Fixed map')}")
            print(f"  Max steps per episode: {self.config.get('max_steps', 1000)}")
            print("\n  Sensor Configuration:")
            if self.config.get('heterogeneous_sensors', False):
                print(f"    Heterogeneous sensors configured")
                for i, sensor in enumerate(self.config.get('agent_sensors', [])):
                    print(f"    Agent {i}: {sensor}")
            else:
                print(f"    Sensor type: {self.config.get('sensor_type', 'camera')}")
                print(f"    Sensor range: {self.config.get('sensor_range', 10)}")
                if self.config.get('sensor_type', 'camera') == 'camera':
                    print(f"    FOV: {self.config.get('sensor_fov', 60)}°")
                print(f"    Sensor rays: {self.config.get('sensor_rays', 30)}")

            print("\n  Reward Configuration:")
            print(f"    Discovery reward: {self.config.get('discovery_reward', 0.1)}")
            print(f"    Collision penalty: {self.config.get('collision_penalty', -1.0)}")
            print(f"    Step penalty: {self.config.get('step_penalty', -0.001)}")
            print(f"    Completion bonus: {self.config.get('completion_bonus', 10.0)}")

            print("\n  PPO Hyperparameters:")
            print(f"    Total timesteps: {self.config['total_timesteps']}")
            print(f"    Learning rate: {self.config.get('learning_rate', 3e-4)}")
            print(f"    Batch size: {self.config.get('batch_size', 64)}")
            print(f"    Steps per update (n_steps): {self.config.get('n_steps', 2048)}")
            print(f"    PPO epochs: {self.config.get('n_epochs', 10)}")
            print(f"    Gamma (discount): {self.config.get('gamma', 0.99)}")
            print(f"    GAE lambda: {self.config.get('gae_lambda', 0.95)}")
            print(f"    Clip range: {self.config.get('clip_range', 0.2)}")
            print(f"    Entropy coefficient: {self.config.get('ent_coef', 0.01)}")

            print("\n  Training Configuration:")
            print(f"    Parallel environments: {self.config.get('n_envs', 1)}")
            print(f"    Evaluation frequency: {self.config.get('eval_freq', 10000)} steps")
            print(f"    Evaluation episodes: {self.config.get('n_eval_episodes', 10)}")
            print(f"    Render during training: {self.config.get('render_training', False)}")
            print(f"    Save path: {self.config['save_path']}")
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
            print("\nRunning quick demo (10000 steps training)...")
            self.config = {
                'num_agents': 1,
                'heterogeneous_sensors': False,
                'grid_width': 15,
                'grid_height': 15,
                'max_steps': 300,
                'randomize_maps': True,
                'sensor_type': 'camera',
                'sensor_range': 7,
                'sensor_fov': 60,
                'sensor_rays': 15,
                'discovery_reward': 0.1,
                'collision_penalty': -1.0,
                'step_penalty': -0.001,
                'completion_bonus': 5.0,
                'total_timesteps': 10000,
                'learning_rate': 3e-4,
                'n_steps': 512,
                'batch_size': 64,
                'n_epochs': 10,
                'gamma': 0.99,
                'gae_lambda': 0.95,
                'clip_range': 0.2,
                'ent_coef': 0.01,
                'n_envs': 2,
                'eval_freq': 2000,
                'n_eval_episodes': 3,
                'render_training': False,
                'save_path': 'models/ppo_demo'
            }

            # Train
            self.train()

            # Test
            self.config['model_path'] = 'models/ppo_demo/best/best_model'
            self.config['test_episodes'] = 3
            self.config['render_test'] = True

            print("\n\nNow testing the trained model...")
            input("Press Enter to continue...")
            self.test()

        elif choice == "4":
            # Multi-agent demo with heterogeneous sensors
            print("\nRunning multi-agent demo (3 agents with different sensors)...")
            self.config = {
                'num_agents': 3,
                'heterogeneous_sensors': True,
                'agent_sensors': [
                    {'type': 'camera', 'range': 8, 'fov': 60, 'rays': 20},
                    {'type': 'camera', 'range': 12, 'fov': 45, 'rays': 30},
                    {'type': 'camera', 'range': 10, 'fov': 90, 'rays': 25}
                ],
                'grid_width': 30,
                'grid_height': 30,
                'max_steps': 1500,
                'randomize_maps': True,
                'discovery_reward': 0.1,
                'collision_penalty': -1.0,
                'step_penalty': -0.001,
                'completion_bonus': 15.0,
                'total_timesteps': 50000,
                'learning_rate': 3e-4,
                'n_steps': 1024,
                'batch_size': 64,
                'n_epochs': 10,
                'gamma': 0.99,
                'gae_lambda': 0.95,
                'clip_range': 0.2,
                'ent_coef': 0.02,
                'n_envs': 2,
                'eval_freq': 5000,
                'n_eval_episodes': 5,
                'render_training': False,
                'save_path': 'models/ppo_multi_demo'
            }

            # Train
            self.train()

            # Test
            self.config['model_path'] = 'models/ppo_multi_demo/best/best_model'
            self.config['test_episodes'] = 3
            self.config['render_test'] = True

            print("\n\nNow testing the trained multi-agent model...")
            input("Press Enter to continue...")
            self.test()

        elif choice == "5":
            print("\nExiting...")
            return
        else:
            print("\nInvalid option. Exiting...")


if __name__ == "__main__":
    trainer = InteractivePPOTrainer()
    trainer.run()