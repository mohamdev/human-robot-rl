import numpy as np

def get_links_kinematics(robot, q, links_list):
    robot.set_qpos(q)
    links_kinematics = np.zeros((len(links_list), 7)) #links_kinematics is a pos & quat (7, n_links) array 
    for i, link in enumerate(links_list):
        links_kinematics[i,:] = np.hstack([link.get_pos().cpu().numpy(), link.get_quat().cpu().numpy()]) # get (pos quat) vector of the curent link
    return links_kinematics[:,:3], links_kinematics[:,3:7] #return a (3, n_links) array of positions and a (4, n_links) array of quaternions