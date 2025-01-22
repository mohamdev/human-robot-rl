import argparse
import os
import pickle
import shutil

from mate_env import MateEnv
from rsl_rl.runners import OnPolicyRunner

import genesis as gs


def get_train_cfg(exp_name, max_iterations):

    train_cfg_dict = {
        "algorithm": {
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.01,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 0.001,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
        "init_member_classes": {},
        "policy": {
            "activation": "elu",
            "actor_hidden_dims": [512, 256, 128],
            "critic_hidden_dims": [512, 256, 128],
            "init_noise_std": 1.0,
        },
        "runner": {
            "algorithm_class_name": "PPO",
            "checkpoint": -1,
            "experiment_name": exp_name,
            "load_run": -1,
            "log_interval": 1,
            "max_iterations": max_iterations,
            "num_steps_per_env": 24,
            "policy_class_name": "ActorCritic",
            "record_interval": -1,
            "resume": False,
            "resume_path": None,
            "run_name": "",
            "runner_class_name": "runner_class_name",
            "save_interval": 100,
        },
        "runner_class_name": "OnPolicyRunner",
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
        "kp": 20.0,
        "kd": 0.5,
        # Termination conditions
        "termination_if_end_effector_z_lower_than": -0.1,
        "termination_if_third_joint_z_lower_than": 0.0,
        # Base pose
        "base_init_pos": [0.0, 0.0, 0.0],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        # Episode settings
        "episode_length_s": 5.0,
        "resampling_time_s": 5.0,  # Keep the target constant throughout the episode
        "action_scale": 0.25,  # Scales actions before applying
        "simulate_action_latency": False,  # No latency for simplicity
        "clip_actions": 3.14,  # Clip actions to stay within joint limits
    }

    obs_cfg = {
        "num_obs": 15,  # Total number of observations 
        #3 target positions, 3 dof angles, 3 dof velocities, 3 previous actions, 3 EE positions
        "obs_scales": {
            "dof_pos": 1.0,  # Scale for joint positions
            "dof_vel": 0.1,  # Scale for joint velocities
            "end_effector_pos": 1.0,  # Scale for end-effector positions
            "target_pos": 1.0,  # Scale for target positions
        },
    }

    reward_cfg = {
        "tracking_sigma": 0.1,  # Higher precision for end-effector tracking
        "reward_scales": {
            "distance_to_target": 10.0,  # Main reward for minimizing distance to the target
            "action_rate": -0.01,  # Penalizes large action changes
            "similar_to_default": -0.1,  # Optional penalty for deviating from default posture
        },
    }

    command_cfg = {
        "num_commands": 3,  # Target position in Cartesian coordinates (x, y, z)
        "x_pos_range": [-0.25, 0.25],  # Range for the x-coordinate
        "y_pos_range": [-0.25, 0.25],  # Range for the y-coordinate
        "z_pos_range": [0.0, 0.5],  # Range for the z-coordinate
    }

    return env_cfg, obs_cfg, reward_cfg, command_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="go2-walking")
    parser.add_argument("-B", "--num_envs", type=int, default=4096)
    parser.add_argument("--max_iterations", type=int, default=100)
    args = parser.parse_args()

    gs.init(logging_level="warning")

    log_dir = f"../logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    train_cfg = get_train_cfg(args.exp_name, args.max_iterations)

    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    env = MateEnv(
        num_envs=args.num_envs, env_cfg=env_cfg, obs_cfg=obs_cfg, reward_cfg=reward_cfg, command_cfg=command_cfg
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device="cuda:0")

    pickle.dump(
        [env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg],
        open(f"{log_dir}/cfgs.pkl", "wb"),
    )

    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    main()
