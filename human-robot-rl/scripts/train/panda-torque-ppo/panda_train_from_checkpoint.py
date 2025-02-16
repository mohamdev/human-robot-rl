import argparse
import os
import pickle
import shutil

from panda_env import PandaEnv
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
            # "actor_hidden_dims": [256, 128],
            # "critic_hidden_dims": [256, 128],
            "init_noise_std": 1.0,
            # "init_noise_std": 0.01,
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
            "save_interval": 100,
        },
        "runner_class_name": "OnPolicyRunner",
        "seed": 1,
    }

    return train_cfg_dict


def get_cfgs():
    env_cfg = {
        "num_actions": 7,  # One action for each of the 3 revolute joints
        "default_joint_angles": {  # Default joint angles in radians
            "joint1": 0.0,
            "joint2": 0.0,
            "joint3": 0.0,
            "joint4": 0.0,
            "joint5": 0.0,
            "joint6": 0.0,
            "joint7": 0.0,
            # "finger_joint1": 0.0,
            # "finger_joint2": 0.0,
        },
        "dof_names": [
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "joint6",
            "joint7",
            # "finger_joint1",
            # "finger_joint2",
        ],

        #Torque bounds
        "force_lower_bound": [-87, -87, -87, -87, -12, -12, -12],
        "force_upper_bound": [ 87,  87,  87,  87,  12,  12,  12],

        "q_lower": [-0.0000001, -0.0000001, -0.0000001, -0.0000001, -0.0000001, -0.00000012, -0.00000012],
        "q_upper": [ 0.0000001,  0.0000001,  0.0000001, 0.00000017,  0.0000001,  0.0000001,  0.0000001],

        # Termination conditions
        "termination_if_end_effector_z_lower_than": 0.15,
        # "termination_if_third_joint_z_lower_than": -10.0,
        # Base pose
        "base_init_pos": [0.0, 0.0, 0.0],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        # Episode settings
        "episode_length_s": 6.0,
        "resampling_time_s": 2.0,  # The target changes every resampling_time_s seconds
        "action_scale": 1.0,  # Scales actions before applying
        "simulate_action_latency": False,  # No latency for simplicity
        "clip_actions": 100.0,  # Clip actions to stay within joint limits
    }

    obs_cfg = {
        "num_obs": 27,  # Total number of observations 
        #3 target positions, 7 dof angles, 7 dof velocities, 7 previous actions, 3 EE positions
        "obs_scales": {
            "dof_pos": 1.0,  # Scale for joint positions
            # "dof_vel": 0.1,  # Scale for joint velocities
            "dof_force": 1.0,  # Scale for joint velocities
            "end_effector_pos": 1.0,  # Scale for end-effector positions
            "target_pos": 1.0,  # Scale for target positions
        },
    }

    reward_cfg = {
        "tracking_sigma": 0.1,  # Higher precision for end-effector tracking
        "reward_scales": {
            "distance_to_target": 1.0,  # Main reward for minimizing distance to the target
            # "reach_target": 1.0,
            # "vel_penalty": 1e-1,
            # "terminal_distance": -1.0,  # Adjust the coefficient as needed
            # "action_rate": 1e-2,  # Penalizes large action changes
            "torques_norm": 1e-3,  # Penalizes large action changes
            "similar_to_default": 1e-3,  # Optional penalty for deviating from default posture
        },
    }

    command_cfg = {
        "num_commands": 3,  # Target position in Cartesian coordinates (x, y, z)
        # "x_pos_range": [8.8000e-02, 9.8000e-02],  # Range for the x-coordinate
        # "y_pos_range": [2.5605e-08, 3.5605e-08],  # Range for the y-coordinate
        # "z_pos_range": [8.6760e-01, 8.6760e-01],  # Range for the z-coordinate

        "x_pos_range": [2.8000e-01, 6.8000e-01],  # Range for the x-coordinate
        "y_pos_range": [2.5605e-01, 6.5605e-01],  # Range for the y-coordinate
        "z_pos_range": [5.6760e-01, 7.6760e-01],  # Range for the z-coordinate

        # "x_pos_range": [-2.8000e-01, 2.8000e-01],  # Range for the x-coordinate
        # "y_pos_range": [-2.5605e-01, 2.5605e-01],  # Range for the y-coordinate
        # "z_pos_range": [5.6760e-01, 7.6760e-01],  # Range for the z-coordinate
    }
    return env_cfg, obs_cfg, reward_cfg, command_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="panda-torque-ppo")
    parser.add_argument("-B", "--num_envs", type=int, default=2800)
    parser.add_argument("--max_iterations", type=int, default=15000)
    args = parser.parse_args()

    gs.init(logging_level="warning")

    log_dir = f"../../logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    train_cfg = get_train_cfg(args.exp_name, args.max_iterations)

    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    env = PandaEnv(
        num_envs=args.num_envs, env_cfg=env_cfg, obs_cfg=obs_cfg, reward_cfg=reward_cfg, command_cfg=command_cfg
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device="cuda:0")
    resume_path = os.path.join("../../logs/ok-policies/", f"model_2800.pt")
    runner.load(resume_path)

    # runner = OnPolicyRunner(env, train_cfg, log_dir, device="cuda:0")

    pickle.dump(
        [env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg],
        open(f"{log_dir}/cfgs.pkl", "wb"),
    )

    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)

if __name__ == "__main__":
    main()
