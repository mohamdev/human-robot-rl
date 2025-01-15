import numpy as np
import genesis as gs

########################## init ##########################
gs.init()

########################## create a scene ##########################
scene = gs.Scene(
    viewer_options= gs.options.ViewerOptions(
        camera_pos    = (0.0, -2, 1.5),
        camera_lookat = (0.0, 0.0, 0.5),
        camera_fov    = 60,
        max_FPS       = 200,
    ),
    show_viewer = False,
    rigid_options=gs.options.RigidOptions(
        enable_joint_limit = False,
    ),
)

########################## entities ##########################
plane = scene.add_entity(
    gs.morphs.Plane(),
)
robot = scene.add_entity(
    gs.morphs.URDF(
        file  = '../models/mate.urdf',
        pos=(0.0, 0.0, 0.0),
        quat=(1.0, 0.0, 0.0, 0.0),
        fixed = True,
        links_to_keep = ['end_effector'])
)

########################## build ##########################
n_envs = 1
scene.build(n_envs=n_envs, env_spacing=(1.0, 1.0))

target_quat = np.tile(np.array([0, 1, 0, 0]), [n_envs, 1]) # pointing downwards
center = np.tile(np.array([0.15, 0.15, 0.15]), [n_envs, 1])
angular_speed = np.random.uniform(-10, 10, n_envs)
r = 0.1

for link in robot.links:
    print(link.name)
ee_link = robot.get_link('end_effector')
q0 = np.zeros(robot.n_qs)
robot.set_qpos(q0)
target_ee_q0 = np.hstack([ee_link.get_pos().cpu().numpy(), ee_link.get_quat().cpu().numpy()])

for i in range(0, 1000):
    # target_pos = np.zeros([n_envs, 3])
    # target_pos[:, 0] = center[:, 0] + np.cos(i/360*np.pi*angular_speed) * r
    # target_pos[:, 1] = center[:, 1] + np.sin(i/360*np.pi*angular_speed) * r
    # target_pos[:, 2] = center[:, 2]
    # target_q = np.hstack([target_pos, target_quat])

    q = robot.inverse_kinematics(
        link     = ee_link,
        pos      = target_ee_q0[:3]
    )

    robot.set_qpos(q)
    scene.step()