import argparse
import os
import pickle
import shutil

from mate_env import MateEnv
import genesis as gs

# Use the new generic runner and SAC algorithm from rsl_rl.
from rsl_rl.runners.runner import Runner
from rsl_rl.algorithms import SAC


def print_callback(runner, stats):
    # This callback prints key statistics every iteration.
    current_iter = stats["current_iteration"]
    if stats["lengths"]:
        avg_reward = sum(stats["returns"]) / len(stats["returns"])
    else:
        avg_reward = 0.0
    print(f"Iteration: {current_iter}, Avg Reward: {avg_reward:.4f}")
    return True  # Return True to continue training

def save_model_callback(runner, stats):
    # Save the model every 50 iterations (adjust as needed)
    if stats["current_iteration"] % 5000 == 0:
        save_path = os.path.join("../../logs/mate-impedance-sac/", f"model_{stats['current_iteration']}.pt")
        runner.save(save_path)
        print(f"Model saved at iteration {stats['current_iteration']} to {save_path}")
    return True


def get_train_cfg(exp_name, max_iterations):
    # SAC hyperparameters have been chosen based on common practice in continuous control.
    # You may want to adjust these based on your environment and desired performance.
    train_cfg_dict = {
        "algorithm": {
            # Learning rates for actor and critic
            "actor_lr": 3e-4,
            "critic_lr": 3e-4,
            # Entropy (temperature) settings
            "alpha": 0.2,
            "alpha_lr": 3e-4,
            # Whether to use separate heads for mean and std (see SAC implementation)
            "chimera": True,
            # Gradient clipping
            "gradient_clip": 100.0,
            # Log standard deviation limits for the actor network
            "log_std_max": 4.0,
            "log_std_min": -20.0,
            # Replay buffer parameters
            "storage_initial_size": 1000,
            "storage_size": 1000000,
            # Target entropy (by default, set to negative of action dimensionality)
            "target_entropy": -7,  # 7 actions → -7
            # Action scaling parameters (set according to your environment's action bounds)
            "action_max": 1000000.0,
            "action_min": -1000000.0,
            # Initial noise standard deviation for action sampling
            "actor_noise_std": 1.0,
        },
        "runner": {
            "checkpoint": -1,
            "experiment_name": exp_name,
            "load_run": -1,
            "log_interval": 1,
            "max_iterations": max_iterations,
            # For off-policy SAC, you typically collect one step per environment per iteration.
            "num_steps_per_env": 1,
            "record_interval": -1,
            "resume": False,
            "resume_path": None,
            "run_name": "",
            "save_interval": 50,
        },
        "seed": 1,
    }
    return train_cfg_dict

def get_cfgs():
    env_cfg = {
        "num_actions": 3,  # One action for each of the 3 revolute joints
        "default_joint_angles": {  # Default joint angles in radians
            "q_1": 0.0,
            "q_2": 0.0,
            "q_3": 0.0,
        },
        "dof_names": [
            "q_1",
            "q_2",
            "q_3",
        ],
        # PD control gains
        "kp": 4500.0,
        "kd": 450.0,
        # Termination conditions
        # "termination_if_end_effector_z_lower_than": -10.0,
        # "termination_if_third_joint_z_lower_than": -10.0,
        # Base pose
        "base_init_pos": [0.0, 0.0, 0.0],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        # Episode settings
        "episode_length_s": 5.0,
        "resampling_time_s": 1.0,  # Keep the target constant throughout the episode
        "action_scale": 1.0,  # Scales actions before applying
        "simulate_action_latency": False,  # No latency for simplicity
        "clip_actions": 10000.0,  # Clip actions to stay within joint limits
    }

    obs_cfg = {
        "num_obs": 12,  # Total number of observations 
        #3 target positions, 3 dof angles, 3 dof velocities, 3 previous actions, 3 EE positions
        "obs_scales": {
            "dof_pos": 1.0,  # Scale for joint positions
            # "dof_vel": 0.1,  # Scale for joint velocities
            "end_effector_pos": 1.0,  # Scale for end-effector positions
            "target_pos": 1.0,  # Scale for target positions
        },
    }

    reward_cfg = {
        "tracking_sigma": 0.1,  # Higher precision for end-effector tracking
        "reward_scales": {
            "distance_to_target": 1.0,  # Main reward for minimizing distance to the target
            # "action_rate": -0.0,  # Penalizes large action changes
            # "similar_to_default": -0.0,  # Optional penalty for deviating from default posture
        },
    }

    command_cfg = {
        "num_commands": 3,  # Target position in Cartesian coordinates (x, y, z)
        "x_pos_range": [0.15, 0.3],  # Range for the x-coordinate
        "y_pos_range": [0.15, 0.3],  # Range for the y-coordinate
        "z_pos_range": [0.15, 0.45],  # Range for the z-coordinate
    }

    return env_cfg, obs_cfg, reward_cfg, command_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="mate-impedance-sac")
    parser.add_argument("-B", "--num_envs", type=int, default=2000)
    parser.add_argument("--max_iterations", type=int, default=500000)
    args = parser.parse_args()

    gs.init(logging_level="warning")

    log_dir = f"../../logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    train_cfg = get_train_cfg(args.exp_name, args.max_iterations)

    # Remove and recreate the log directory if it exists.
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    # Create the environment.
    env = MateEnv(
        num_envs=args.num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
    )

    # Instantiate the SAC agent using the hyperparameters from our configuration.
    agent = SAC(env, device="cuda:0", **train_cfg["algorithm"])

    learn_callbacks = [print_callback, save_model_callback]
    # Create the runner; note that num_steps_per_env is set to 1 for off-policy algorithms like SAC.
    runner = Runner(
        env,
        agent,
        device="cuda:0",
        num_steps_per_env=train_cfg["runner"]["num_steps_per_env"],
        evaluation_cb=[],  # You can add evaluation callbacks here if desired.
        learn_cb=learn_callbacks,    # You can add learning callbacks (e.g., logging, saving) here.
    )

    # Optionally, save your configuration for later reference.
    pickle.dump(
        [env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg],
        open(f"{log_dir}/cfgs.pkl", "wb"),
    )

    
    # Start the learning process.
    runner.learn(iterations=args.max_iterations)


if __name__ == "__main__":
    main()
