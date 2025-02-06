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
            "learning_rate": 0.0005,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 3,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 0.5,
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
            "num_steps_per_env": 100,
            "policy_class_name": "ActorCritic",
            "record_interval": -1,
            "resume": False,
            "resume_path": None,
            "run_name": "",
            "runner_class_name": "runner_class_name",
            "save_interval": 50,
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
        "kp": 4500.0,
        "kd": 450.0,
        # Termination conditions
        "termination_if_end_effector_z_lower_than": -10.0,
        "termination_if_third_joint_z_lower_than": 0.0,
        # Base pose
        "base_init_pos": [0.0, 0.0, 0.0],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        # Episode settings
        "resampling_time_s": 0.025,  # Update target every step (match control timestep)
        "episode_length_s": 5.0,    # Time to complete the line
        "action_scale": 1.0,  # Scales actions before applying
        "simulate_action_latency": False,  # No latency for simplicity
        "clip_actions": 10000.0,  # Clip actions to stay within joint limits
    }

    obs_cfg = {
        "num_obs": 15,  # Total number of observations 
        #3 target positions, 3 dof angles, 3 dof velocities, 3 previous actions, 3 EE positions
        "obs_scales": {
            "dof_pos": 1.0,  # Scale for joint positions
            # "dof_vel": 0.1,  # Scale for joint velocities
            "end_effector_pos": 1.0,  # Scale for end-effector positions
            "target_pos": 1.0,  # Scale for target positions
            "line_direction": 1.0,
        },
    }

    reward_cfg = {
        "tracking_sigma": 0.1,
        "reward_scales": {
            "distance_to_target": 1.0,
            "path_progress": 0.5,  # New component
        }
    }

    command_cfg = {
        "num_commands": 6,  # Start (x1,y1,z1) and end (x2,y2,z2) of the line
        "line_length_range": [0.2, 0.2],  # Length of generated lines
        "line_center_range": {  # Area where lines are generated
            "x": [0.2, 0.3],
            "y": [0.2, 0.3],
            "z": [0.2, 0.4]
        },
    }

    return env_cfg, obs_cfg, reward_cfg, command_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="mate-reach-dist")
    parser.add_argument("-B", "--num_envs", type=int, default=5000)
    parser.add_argument("--max_iterations", type=int, default=500)
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
