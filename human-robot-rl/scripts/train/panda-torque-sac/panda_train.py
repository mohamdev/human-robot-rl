import argparse
import os
import pickle
import shutil

from panda_env import PandaEnv
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
    if stats["current_iteration"] % 50 == 0:
        save_path = os.path.join("../../logs/panda-reach-torque-sac/", f"model_{stats['current_iteration']}.pt")
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
            "gradient_clip": 1.0,
            # Log standard deviation limits for the actor network
            "log_std_max": 4.0,
            "log_std_min": -20.0,
            # Replay buffer parameters
            "storage_initial_size": 1000,
            "storage_size": 1000000,
            # Target entropy (by default, set to negative of action dimensionality)
            "target_entropy": -7,  # 7 actions → -7
            # Action scaling parameters (set according to your environment's action bounds)
            "action_max": 100.0,
            "action_min": -100.0,
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
    # Environment configuration (same as before)
    env_cfg = {
        "num_actions": 7,  # 7 joints
        "default_joint_angles": {
            "joint1": 0.0,
            "joint2": 0.0,
            "joint3": 0.0,
            "joint4": 0.0,
            "joint5": 0.0,
            "joint6": 0.0,
            "joint7": 0.0,
        },
        "dof_names": [
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "joint6",
            "joint7",
        ],
        "q_lower": [-1e-7] * 7,
        "q_upper": [1e-7] * 7,
        "termination_if_end_effector_z_lower_than": 0.15,
        "base_init_pos": [0.0, 0.0, 0.0],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "episode_length_s": 5.0,
        "resampling_time_s": 5.0,
        "action_scale": 1.0,
        "simulate_action_latency": False,
        "clip_actions": 10000.0,
    }

    obs_cfg = {
        "num_obs": 27,
        "obs_scales": {
            "dof_pos": 1.0,
            "dof_force": 1.0,
            "end_effector_pos": 1.0,
            "target_pos": 1.0,
        },
    }

    reward_cfg = {
        "tracking_sigma": 0.1,
        "reward_scales": {
            "vel_penalty": 1e-1,
            "action_rate": 1e-2,
            "similar_to_default": 10.0,
        },
    }

    command_cfg = {
        "num_commands": 3,  # For target position (x, y, z)
        "x_pos_range": [8.8e-2, 9.8e-2],
        "y_pos_range": [2.56e-8, 3.56e-8],
        "z_pos_range": [8.676e-1, 8.676e-1],
    }
    return env_cfg, obs_cfg, reward_cfg, command_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="panda-reach-torque-sac")
    parser.add_argument("-B", "--num_envs", type=int, default=2000)
    parser.add_argument("--max_iterations", type=int, default=15000)
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
    env = PandaEnv(
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
