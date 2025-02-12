import argparse
import os
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt  # for plotting

from panda_env import PandaEnv
from rsl_rl.runners import OnPolicyRunner
import genesis as gs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="panda-torque-ppo")
    parser.add_argument("--ckpt", type=int, default=2800)
    args = parser.parse_args()

    # Initialize Genesis
    gs.init()

    # Load configurations
    log_dir = f"../../logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(open(f"{log_dir}/cfgs.pkl", "rb"))
    reward_cfg["reward_scales"] = {}
    env_cfg["episode_length_s"] = 20.0
    env_cfg["resampling_time_s"] = 2.0 

    # Initialize environment
    env = PandaEnv(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=True,
    )

    # Load trained policy
    runner = OnPolicyRunner(env, train_cfg, log_dir, device="cuda:0")
    resume_path = os.path.join(log_dir, f"model_{args.ckpt}.pt")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device="cuda:0")

    # Reset environment and initialize storage lists
    obs, _ = env.reset()
    joint_angles_history = []  # each element will be an array of joint angles at one time step
    joint_torques_history = []  # each element will be an array of joint torques commanded at one time step

    done = False
    with torch.no_grad():
        while not done:
            # Get the actions from the policy (assumed to be joint torques)
            actions = policy(obs)
            # Save the joint torques (convert to numpy and flatten)
            joint_torques_history.append(actions.cpu().numpy().flatten())

            # Get and save the joint angles (assumed available from the robot object)
            joint_angles = env.robot.get_dofs_position()
            joint_angles_history.append(joint_angles.cpu().numpy().flatten())

            # Step the environment
            obs, _, rews, dones, infos = env.step(actions)

            # Update the target sphere position (if applicable)
            target_position = env.commands.cpu().numpy()
            env.target_sphere.set_pos(target_position)

            # End the episode if the environment signals done
            if dones[0]:
                print("Episode finished.")
                done = True

    # Convert histories to numpy arrays for easier indexing
    joint_angles_history = np.array(joint_angles_history)  # shape: (timesteps, num_joints)
    joint_torques_history = np.array(joint_torques_history)  # shape: (timesteps, num_joints)
    num_joints = 7

    # Try to obtain joint angle bounds from the robot; otherwise, use typical Panda limits.
    try:
        joint_angle_lower_bounds = env.robot.get_dofs_lower_limits().cpu().numpy().flatten()
        joint_angle_upper_bounds = env.robot.get_dofs_upper_limits().cpu().numpy().flatten()
    except AttributeError:
        # These are example limits for the Franka Emika Panda arm.
        joint_angle_lower_bounds = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
        joint_angle_upper_bounds = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])

    # Similarly, try to obtain joint torque bounds; if not available, use dummy values.
    try:
        joint_torque_lower_bounds = env.robot.get_joint_torque_lower_bounds().cpu().numpy().flatten()
        joint_torque_upper_bounds = env.robot.get_joint_torque_upper_bounds().cpu().numpy().flatten()
    except AttributeError:
        # These dummy values are examples and may need adjustment.
        joint_torque_lower_bounds = np.array([-87, -87, -87, -87, -12, -12, -12])
        joint_torque_upper_bounds = np.array([87, 87, 87, 87, 12, 12, 12])

    # Plot joint angles (each joint in its own subplot)
    fig_angles, axes_angles = plt.subplots(num_joints, 1, figsize=(10, 2 * num_joints), sharex=True)
    if num_joints == 1:
        axes_angles = [axes_angles]
    for i in range(num_joints):
        axes_angles[i].plot(joint_angles_history[:, i], label=f'Joint {i+1} Angle')
        axes_angles[i].axhline(y=joint_angle_lower_bounds[i], color='r', linestyle='--',
                                 label='Lower Bound' if i == 0 else "")
        axes_angles[i].axhline(y=joint_angle_upper_bounds[i], color='g', linestyle='--',
                                 label='Upper Bound' if i == 0 else "")
        axes_angles[i].set_ylabel("Angle (rad)")
        axes_angles[i].legend(loc='upper right')
    axes_angles[-1].set_xlabel("Time step")
    fig_angles.suptitle("Joint Angles Over One Episode")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

    # Plot joint torques (each joint in its own subplot)
    fig_torques, axes_torques = plt.subplots(num_joints, 1, figsize=(10, 2 * num_joints), sharex=True)
    if num_joints == 1:
        axes_torques = [axes_torques]
    for i in range(num_joints):
        axes_torques[i].plot(joint_torques_history[:, i], label=f'Joint {i+1} Torque')
        axes_torques[i].axhline(y=joint_torque_lower_bounds[i], color='r', linestyle='--',
                                  label='Lower Bound' if i == 0 else "")
        axes_torques[i].axhline(y=joint_torque_upper_bounds[i], color='g', linestyle='--',
                                  label='Upper Bound' if i == 0 else "")
        axes_torques[i].set_ylabel("Torque (Nm)")
        axes_torques[i].legend(loc='upper right')
    axes_torques[-1].set_xlabel("Time step")
    fig_torques.suptitle("Joint Torques Over One Episode")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


if __name__ == "__main__":
    main()
