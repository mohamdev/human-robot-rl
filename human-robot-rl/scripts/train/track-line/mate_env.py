import torch
import math
import genesis as gs
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat


def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower


class MateEnv:
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=False, device="cuda"):
        self.device = torch.device(device)

        self.num_envs = num_envs
        self.num_obs = obs_cfg["num_obs"]
        self.num_privileged_obs = None
        self.num_actions = env_cfg["num_actions"]
        self.num_commands = command_cfg["num_commands"]
        self.simulate_action_latency = False 

        # self.simulate_action_latency = True  # there is a 1 step latency on real robot
        self.dt = 0.025  # control frequence on real robot is 50hz
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
            gs.morphs.URDF(
                file="../../models/mate.urdf",
                pos=self.base_init_pos.cpu().numpy(),
                quat=self.base_init_quat.cpu().numpy(),
                fixed = True,
                links_to_keep = ['end_effector']
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

        # PD control parameters
        self.robot.set_dofs_kp([self.env_cfg["kp"]] * self.num_actions, self.motor_dofs)
        self.robot.set_dofs_kv([self.env_cfg["kd"]] * self.num_actions, self.motor_dofs)

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
        self.dof_vel = torch.zeros_like(self.dof_pos)
        self.last_dof_vel = torch.zeros_like(self.dof_pos)
        self.default_dof_pos = torch.tensor(
            [env_cfg["default_joint_angles"][name] for name in env_cfg["dof_names"]],
            device=self.device,
            dtype=gs.tc_float,
        )
        self.extras = {}

    def _resample_commands(self, envs_idx):
        n_envs = len(envs_idx)
        # Sample line start points within workspace
        start = torch.zeros((n_envs, 3), device=self.device)
        start[:, 0] = gs_rand_float(*self.command_cfg["x_pos_range"], (n_envs,), self.device)
        start[:, 1] = gs_rand_float(*self.command_cfg["y_pos_range"], (n_envs,), self.device)
        start[:, 2] = gs_rand_float(*self.command_cfg["z_pos_range"], (n_envs,), self.device)
        
        # Sample line direction and length
        theta = torch.rand(n_envs, device=self.device) * 2 * torch.pi  # Azimuth angle
        phi = torch.rand(n_envs, device=self.device) * torch.pi / 2  # Polar angle (limit to XY plane)
        direction = torch.stack([torch.cos(theta), torch.sin(theta), torch.zeros_like(theta)], dim=1)
        length = gs_rand_float(*self.command_cfg["line_length_range"], (n_envs,), self.device)
        
        # Calculate end point
        end = start + direction * length.view(-1, 1)
        
        # Store both start and end in commands buffer
        self.commands[envs_idx, 0:3] = start
        self.commands[envs_idx, 3:6] = end

    def step(self, actions):
        # Clip and scale actions
        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
        exec_actions = self.last_actions if self.simulate_action_latency else self.actions
        target_dof_pos = exec_actions * self.env_cfg["action_scale"] + self.default_dof_pos
        self.robot.control_dofs_position(target_dof_pos, self.motor_dofs)

        # Get end-effector position
        self.ee_link = self.robot.get_link("end_effector")
        ee_pos = self.ee_link.get_pos()
        
        # Update visualization
        self.ee_sphere.set_pos(ee_pos)
        self.scene.step()

        # Update episode step count
        self.episode_length_buf += 1

        # Update joint states
        self.dof_pos[:] = self.robot.get_dofs_position(self.motor_dofs)
        self.dof_vel[:] = self.robot.get_dofs_velocity(self.motor_dofs)

        # Calculate current target along the line
        progress = self.episode_length_buf.float() / self.max_episode_length
        start = self.commands[:, 0:3]
        end = self.commands[:, 3:6]
        current_target = start + (end - start) * progress.unsqueeze(-1)
        
        # Update target visualization (first env only)
        self.target_sphere.set_pos(current_target[0].unsqueeze(0))

        # Check termination conditions
        self.reset_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= ee_pos[:, 2] < self.env_cfg["termination_if_end_effector_z_lower_than"]
        
        # Add line deviation termination
        line_vec = end - start
        ee_vec = ee_pos - start
        t = torch.sum(ee_vec * line_vec, dim=1) / torch.sum(line_vec**2, dim=1)
        closest_point = start + t.unsqueeze(-1) * line_vec
        deviation = torch.norm(ee_pos - closest_point, dim=1)
        self.reset_buf |= deviation > 0.1  # Terminate if >10cm deviation

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
        line_direction = (end - start) * self.obs_scales["line_direction"]
        self.obs_buf = torch.cat(
            [
                self.dof_pos * self.obs_scales["dof_pos"],          # Joint positions
                ee_pos * self.obs_scales["end_effector_pos"],       # End-effector position
                current_target * self.obs_scales["target_pos"],     # Current target position
                line_direction,                                     # Line direction vector
                self.actions,                                       # Previous actions
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
        self.ee_link = self.robot.get_link("end_effector")
        ee_pos = self.ee_link.get_pos()
        target_pos = self.commands  # Target positions from commands
        distance = torch.sqrt(torch.sum((ee_pos - target_pos) ** 2, dim=1))
        # return torch.exp(-distance / self.reward_cfg["tracking_sigma"])  # Exponential decay reward
        return -distance*1.0 + 0.0*torch.exp(-distance / self.reward_cfg["tracking_sigma"])  # Exponential decay reward
        # return -distance  # Stronger linear penalty

    def _reward_action_rate(self):
        # Penalize large changes in consecutive actions
        return -torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    # def _reward_similar_to_default(self):
    #     # Penalize joint configurations far from the default pose
    #     return -torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1)

