import torch
import math
import genesis as gs
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat
import numpy as np

def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower


class PandaEnv:
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=False, device="cuda"):
        self.device = torch.device(device)

        self.num_envs = num_envs
        self.num_obs = obs_cfg["num_obs"]
        self.num_privileged_obs = None
        self.num_actions = env_cfg["num_actions"]
        self.num_commands = command_cfg["num_commands"]
        self.simulate_action_latency = False 

        # self.simulate_action_latency = True  # there is a 1 step latency on real robot
        self.dt = 0.01  # control frequence on real robot is 100hz
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg

        self.obs_scales = obs_cfg["obs_scales"]
        self.reward_scales = reward_cfg["reward_scales"]

        # create scene
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(0.5 / self.dt),
                camera_pos=(2.0, 0.0, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=60,
            ),
            vis_options=gs.options.VisOptions(n_rendered_envs=1),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
            ),
            show_viewer=show_viewer,
        )

        # add plain
        self.scene.add_entity(
        gs.morphs.Plane(),
        )

        # add robot
        self.base_init_pos = torch.tensor(self.env_cfg["base_init_pos"], device=self.device)
        self.base_init_quat = torch.tensor(self.env_cfg["base_init_quat"], device=self.device)
        self.inv_base_init_quat = inv_quat(self.base_init_quat)
        self.robot = self.scene.add_entity(
                                            gs.morphs.MJCF(
                                                file  = 'xml/franka_emika_panda/panda.xml',
                                            ),
                                        )

        # Create a sphere to represent the end-effector position
        self.ee_sphere = self.scene.add_entity(gs.morphs.Sphere(
            pos=(0.0, 0.0, 0.0),  # Initial position
            radius=0.01,          # Small radius for the sphere
            visualization=True,
            collision=False,
        ))

        # Create a sphere to represent the target
        self.target_sphere = self.scene.add_entity(gs.morphs.Sphere(
            pos=(0.0, 0.0, 0.0),  # Initial position
            radius=0.02,          # Small radius for the sphere
            visualization=True,
            collision=False,
        ))

        # build
        self.scene.build(n_envs=num_envs)

        # names to indices
        self.motor_dofs = [self.robot.get_joint(name).dof_idx_local for name in self.env_cfg["dof_names"]]

        # self.robot.set_dofs_force_range(
        #     lower          = np.array(self.env_cfg["force_lower_bound"]),
        #     upper          = np.array(self.env_cfg["force_upper_bound"]),
        #     dofs_idx_local = self.motor_dofs,
        # )

        # prepare reward functions and multiply reward scales by dt
        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)

        # Initialize buffers
        self.obs_buf = torch.zeros((num_envs, self.num_obs), device=self.device, dtype=gs.tc_float)
        self.rew_buf = torch.zeros((num_envs,), device=self.device, dtype=gs.tc_float)
        self.reset_buf = torch.ones((num_envs,), device=self.device, dtype=gs.tc_int)
        self.episode_length_buf = torch.zeros((num_envs,), device=self.device, dtype=gs.tc_int)
        self.commands = torch.zeros((num_envs, self.num_commands), device=self.device, dtype=gs.tc_float)
        self.actions = torch.zeros((num_envs, self.num_actions), device=self.device, dtype=gs.tc_float)
        self.last_actions = torch.zeros((num_envs, self.num_actions), device=self.device, dtype=gs.tc_float)
        self.dof_pos = torch.zeros((num_envs, self.num_actions), device=self.device, dtype=gs.tc_float)
        self.dof_force = torch.zeros_like(self.dof_pos)
        self.dof_vel = torch.zeros_like(self.dof_pos)
        self.last_dof_vel = torch.zeros_like(self.dof_pos)
        self.default_dof_pos = torch.tensor(
            [env_cfg["default_joint_angles"][name] for name in env_cfg["dof_names"]],
            device=self.device,
            dtype=gs.tc_float,
        )
        self.extras = {}

    def _resample_commands(self, envs_idx):
        # print("resample commands")
        # Sample random target positions for the end-effector within the specified ranges
        self.commands[envs_idx, 0] = gs_rand_float(*self.command_cfg["x_pos_range"], (len(envs_idx),), self.device)
        self.commands[envs_idx, 1] = gs_rand_float(*self.command_cfg["y_pos_range"], (len(envs_idx),), self.device)
        self.commands[envs_idx, 2] = gs_rand_float(*self.command_cfg["z_pos_range"], (len(envs_idx),), self.device)

    def step(self, actions):
        # print("step")
        # Clip and scale actions
        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
        exec_actions = self.last_actions if self.simulate_action_latency else self.actions
        self.robot.control_dofs_position(exec_actions, self.motor_dofs)

        # Get end-effector position
        self.ee_link = self.robot.get_link("hand")
        ee_pos = self.ee_link.get_pos()
        
        # Update the position of the end-effector sphere
        self.ee_sphere.set_pos(ee_pos)
        # print("EE Position:", ee_pos.cpu().numpy())
        # print("Sphere Position:", self.ee_sphere.get_pos().cpu().numpy())
        self.scene.step()

        # Update episode step count
        self.episode_length_buf += 1

        # Update joint states
        self.dof_pos[:] = self.robot.get_dofs_position(self.motor_dofs)
        self.dof_vel[:] = self.robot.get_dofs_velocity(self.motor_dofs)
        self.dof_force[:] = self.robot.get_dofs_force(self.motor_dofs)

        # Update commands if necessary
        envs_idx = (
            (self.episode_length_buf % int(self.env_cfg["resampling_time_s"] / self.dt) == 0)
            .nonzero(as_tuple=False)
            .flatten()
        )
        # print("envs idx:", envs_idx)
        self._resample_commands(envs_idx)
        
        # Check termination conditions
        # print("self.episode_length_buf > self.max_episode_length:", self.episode_length_buf > self.max_episode_length)
        # print("max_episode_length:", self.max_episode_length)
        self.reset_buf = self.episode_length_buf > self.max_episode_length
        # print("self.episode_length_buf:", self.episode_length_buf)
        # self.reset_buf |= self.dof_pos[:, 2] < self.env_cfg["termination_if_third_joint_z_lower_than"]
        # print("self.reset_buff:", self.reset_buf)

        # Reset environments
        time_out_idx = (self.episode_length_buf > self.max_episode_length).nonzero(as_tuple=False).flatten()
        self.extras["time_outs"] = torch.zeros_like(self.reset_buf, device=self.device, dtype=gs.tc_float)
        self.extras["time_outs"][time_out_idx] = 1.0
        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).flatten())

        # Compute rewards
        self.rew_buf[:] = 0.0
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew

        # Compute observations
        self.obs_buf = torch.cat(
            [
                self.dof_pos * self.obs_scales["dof_pos"],  # Joint positions
                # self.dof_vel * self.obs_scales["dof_vel"],  # Joint velocities
                self.dof_force * self.obs_scales["dof_force"],  # Joint torques
                ee_pos * self.obs_scales["end_effector_pos"],  # End-effector position
                self.commands * self.obs_scales["target_pos"],  # Target position
                self.actions,  # Previous actions
            ],
            axis=-1,
        )

        # Store last actions and velocities
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]

        return self.obs_buf, None, self.rew_buf, self.reset_buf, self.extras


    def get_observations(self):
        return self.obs_buf

    def get_privileged_observations(self):
        return None

    def reset_idx(self, envs_idx):
        if len(envs_idx) == 0:
            return

        # Reset DOFs
        self.dof_pos[envs_idx] = self.default_dof_pos
        self.dof_vel[envs_idx] = 0.0
        self.robot.set_dofs_position(
            position=self.dof_pos[envs_idx],
            dofs_idx_local=self.motor_dofs,
            zero_velocity=True,
            envs_idx=envs_idx,
        )
        
        # Reset buffers
        self.dof_force[envs_idx] = 0.0
        self.last_actions[envs_idx] = 0.0
        self.last_dof_vel[envs_idx] = 0.0
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx] = True

        # Reinitialize commands (new targets for end-effector)
        self._resample_commands(envs_idx)

        # Reset rewards and extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][envs_idx]).item() / self.env_cfg["episode_length_s"]
            )
            self.episode_sums[key][envs_idx] = 0.0

    def reset(self):
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        return self.obs_buf, None

    # ------------ reward functions ----------------
    def _reward_distance_to_target(self):
        # Reward for minimizing the distance between the end-effector and the target
        self.ee_link = self.robot.get_link("hand")
        ee_pos = self.ee_link.get_pos()
        target_pos = self.commands  # Target positions from commands
        distance = torch.sqrt(torch.sum((ee_pos - target_pos) ** 2, dim=1))
        # return torch.exp(-distance / self.reward_cfg["tracking_sigma"])  # Exponential decay reward
        return -distance**2 + 1.0*torch.exp(-distance / self.reward_cfg["tracking_sigma"])  # Exponential decay reward
        # return -distance**2  # Stronger linear penalty

    def _reward_reach_target(self):
        self.ee_link = self.robot.get_link("hand")
        ee_pos = self.ee_link.get_pos()
        target_pos = self.commands  # Target positions from commands
        distance = torch.sqrt(torch.sum((ee_pos - target_pos) ** 2, dim=1))
        # Use torch.where to return a tensor of rewards for all environments
        reward = torch.where(
            distance < 0.05,
            torch.tensor(10.0, device=distance.device),
            torch.tensor(0.0, device=distance.device)
        )
        return reward

    def _reward_vel_penalty(self):
        # Penalize high velocities
        vel = self.robot.get_dofs_velocity(dofs_idx_local=self.motor_dofs)
        vel_norm = torch.sqrt(torch.sum(vel ** 2, dim=1))
        return -vel_norm

    # def _reward_action_rate(self):
    #     # Penalize large changes in consecutive actions
    #     distance = torch.sum(torch.square(self.last_actions - self.actions), dim=1)
    #     print("distance=",distance)
    #     return torch.exp(-distance)

    # def _reward_similar_to_default(self):
    #     # Penalize joint configurations far from the default pose
    #     return -torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1)

