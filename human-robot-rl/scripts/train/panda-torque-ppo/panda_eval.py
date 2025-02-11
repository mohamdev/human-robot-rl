import argparse
import os
import pickle
import numpy as np
import torch
from panda_env import PandaEnv
from rsl_rl.runners import OnPolicyRunner

import genesis as gs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="panda-torque-ppo")
    parser.add_argument("--ckpt", type=int, default=1000)
    args = parser.parse_args()

    # Initialize Genesis
    gs.init()

    # Load configurations
    log_dir = f"../../logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(open(f"{log_dir}/cfgs.pkl", "rb"))
    reward_cfg["reward_scales"] = {}
    env_cfg["episode_length_s"] = 6.0
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

    # Reset environment
    obs, _ = env.reset()

    res=0
    k=0
    prev_ee_position=0
    prev_residue=0
    # Evaluate the policy
    with torch.no_grad():
        while True:
            # Get actions from the policy
            actions = policy(obs)
            print("joint angles:", actions)
            obs, _, rews, dones, infos = env.step(actions)
            # print("actions:", actions)

            # Update the target sphere position
            target_position = env.commands.cpu().numpy()  # Get target position as a 2D array
            # print("target position:", target_position)
            # print("episode_length_s:", env_cfg["episode_length_s"])
            # print("resampling_time_s:", env_cfg["resampling_time_s"])
            env.target_sphere.set_pos(target_position)    # Update sphere position

            # # Get end-effector position as a NumPy array
            # ee_position = (env.lfinger_link.get_pos().cpu().numpy() + env.rfinger_link.get_pos().cpu().numpy())/2.0
            # env.ee_sphere.set_pos(ee_position)

            # Compute the residue using NumPy
            # residue = np.sqrt(np.sum((ee_position - target_position) ** 2, axis=1))
            # print("residue =", residue)

            # Handle resets
            if dones[0]:  # If the environment resets
                print("reset")

                obs, _ = env.reset()
    print("residue = ", res/k)

if __name__ == "__main__":
    main()
