import torch
import genesis as gs

gs.init(backend=gs.gpu)

scene = gs.Scene(
    show_viewer   = False,
    rigid_options = gs.options.RigidOptions(
        dt                = 0.01,
    ),
)

plane = scene.add_entity(
    gs.morphs.Plane(),
)

franka = scene.add_entity(
    gs.morphs.MJCF(file='xml/franka_emika_panda/panda.xml'),
)

# cam = scene.add_camera(
#     res    = (1920, 1080),
#     pos    = (6.5, 0.0, 2.5),
#     lookat = (0, 0, 0.5),
#     fov    = 60,
#     GUI    = False,
# )

scene.build(n_envs=10000)

# control all the robots
franka.control_dofs_position(
    torch.tile(
        torch.tensor([0, 0, 0, -1.0, 0, 0, 0, 0.02, 0.02], device=gs.device), (10000, 1)
    ),
)

# start camera recording. Once this is started, all the rgb images rendered will be recorded internally
# cam.start_recording()

import numpy as np

for i in range(1000):
    scene.step()

    # # change camera position
    # cam.set_pose(
    #     pos    = (3.0 * np.sin(i / 60), 3.0 * np.cos(i / 60), 2.5),
    #     lookat = (0, 0, 0.5),
    # )
    
    # cam.render()

# stop recording and save video. If `filename` is not specified, a name will be auto-generated using the caller file name.
# cam.stop_recording(save_to_filename='video_frankazz.mp4', fps=30)