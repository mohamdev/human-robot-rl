import numpy as np
import genesis as gs
import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from utils.genesis_utils import get_links_kinematics

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
    show_viewer = True,
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
        file  = '../../models/mate.urdf',
        pos=(0.0, 0.0, 0.0),
        quat=(1.0, 0.0, 0.0, 0.0),
        fixed = True,
        links_to_keep = ['end_effector'])
)

# Create a sphere to represent the end-effector position
ee_sphere = scene.add_entity(gs.morphs.Sphere(
    pos=(0.0, 0.0, 0.0),  # Initial position
    radius=0.01,          # Small radius for the sphere
    visualization=True,
    collision=False,
))


########################## build ##########################
scene.build()

motors_dof = np.arange(robot.n_qs)

# set control gains
# Note: the following values are tuned for achieving best behavior with robot
# Typically, each new robot would have a different set of parameters.
# Sometimes high-quality URDF or XML file would also provide this and will be parsed.
robot.set_dofs_kp(
    np.array([4500, 4500, 4500]),
)
robot.set_dofs_kv(
    np.array([450, 450, 450]),
)
robot.set_dofs_force_range(
    np.array([-87, -87, -87]),
    np.array([ 87,  87,  87]),
)

links_list = [robot.get_link('end_effector')] # get the end-effector link
qi = np.zeros(robot.n_qs)
qf = np.array([0.5, 0.8, -2.0])

pos_eei, quat_eei = get_links_kinematics(robot, qi, links_list)
pos_eef, _ = get_links_kinematics(robot, qf, links_list)

robot.set_qpos(qi)

path = robot.plan_path(
    qpos_goal     = qf,
    num_waypoints = 2000, # 2s duration
)

# execute the planned path
for waypoint in path:
    robot.control_dofs_position(waypoint)
    ee_link = robot.get_link("end_effector")
    ee_pos = ee_link.get_pos()
    
    # Update the position of the end-effector sphere
    ee_sphere.set_pos(ee_pos)
    scene.step()

#now go back to initial q, without path planning
robot.control_dofs_position(qi, motors_dof)
for i in range(100):
    scene.step()

#now reach random configurations without path planning
for n_targets in range(4):
    random_q = np.random.uniform(-np.pi/4, np.pi/2, size=(robot.n_qs,))
    robot.control_dofs_position(random_q, motors_dof)
    for i in range(100):
        scene.step()


