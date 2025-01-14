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
        file  = '../models/mate.urdf',
        pos=(0.0, 0.3, 1.0),
        quat=(1.0, 0.0, 0.0, 0.0),
        fixed = True),
)

print("n_qs:", robot.n_qs)
print("n_links:", robot.n_links)
print("n_joints:", robot.n_joints)
print("n_dofs:", robot.n_dofs)

########################## build ##########################
scene.build()

# start camera recording. Once this is started, all the rgb images rendered will be recorded internally
q = np.zeros(robot.n_qs)
robot.set_qpos(q)

k=0
for i in range(10):
    q = np.zeros(robot.n_qs)
    robot.set_qpos(q)

    scene.step()
    k=i

for i in range(100):
    # cam.render()
    scene.step()

import numpy as np
for i in range(350):
    scene.step()