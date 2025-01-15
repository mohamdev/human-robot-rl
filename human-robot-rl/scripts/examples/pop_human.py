import numpy as np

import genesis as gs

########################## init ##########################
gs.init(backend=gs.gpu)


########################## create a scene ##########################
scene = gs.Scene(
    viewer_options = gs.options.ViewerOptions(
        camera_pos    = (0, -3.5, 2.5),
        camera_lookat = (0.0, 0.0, 0.5),
        camera_fov    = 60,
        res           = (960, 640),
        max_FPS       = 200,
    ),
    sim_options = gs.options.SimOptions(
        dt = 0.01,
    ),
    show_viewer = False,
)

########################## entities ##########################
plane = scene.add_entity(
    gs.morphs.Plane(),
)
robot = scene.add_entity(
    gs.morphs.URDF(
        file  = '../models/human.urdf',
    ),
)

# cam = scene.add_camera(
#     res    = (1280, 720),
#     pos    = (4.5, 0.0, 3.5),
#     lookat = (0, 0, 0.5),
#     fov    = 30,
#     GUI    = False,
# )


print("n_qs:", robot.n_qs)
print("n_links:", robot.n_links)
print("n_joints:", robot.n_joints)
print("n_dofs:", robot.n_dofs)
# for joint in robot.joints:
#     print(joint)

########################## build ##########################
scene.build()

# start camera recording. Once this is started, all the rgb images rendered will be recorded internally
# cam.start_recording()
pos = np.array([0.0, 0.3, 1.0])
quat = np.array([1.0, 0.0, 0.0, 0.0])
# robot.gravity_compensation = 1
q = np.zeros(robot.n_qs)
q[:3] = pos  # Set position
q[3:7] = quat  # Set orientation quaternion
robot.set_qpos(q)

k=0
for i in range(120):
    q = np.zeros(robot.n_qs)
    q[:3] = pos  # Set position
    q[3:7] = quat  # Set orientation quaternion
    robot.set_qpos(q)
    # print("robot qpos:", robot.get_qpos())

    # change camera position
    # cam.set_pose(
    #     pos    = (4.0 * np.sin((i) / 60), 4.0 * np.cos((i) / 60), 3.5),
    #     lookat = (0, 0, 0.5),
    # )

    # cam.render()
    scene.step()
    k=i



for i in range(100):
    # cam.render()
    scene.step()

import numpy as np
for i in range(350):
    scene.step()

    # # change camera position
    # cam.set_pose(
    #     pos    = (4.0 * np.sin((i+k) / 60), 4.0 * np.cos((i+k) / 60), 3.5),
    #     lookat = (0, 0, 0.5),
    # )
    
    # cam.render()



# cam.stop_recording(save_to_filename='video.mp4', fps=60)