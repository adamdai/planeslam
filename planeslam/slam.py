"""SLAM functions

"""

import numpy as np
from copy import deepcopy
from graphslam.graph import Graph
from graphslam.vertex import Vertex
from graphslam.edge.edge_odometry import EdgeOdometry
from graphslam.pose.se3 import PoseSE3

from planeslam.scan import pc_to_scan
from planeslam.geometry.util import quat_to_R
from planeslam.registration import robust_GN_register


def generate_map(traj, scans, dist_thresh=7.5, p2p_thresh=5.0, area_thresh=5.0, fuse_thresh=2.0):
    """Generate map from trajectory and scans by transforming scans and merging them together

    Parameters
    ----------
    traj : list of tuples (R,t)
        Sequence of poses
    scans : list of Scan
        Sequence of scans

    Returns
    -------
    map : Scan
        Merged map
    
    """
    scans_transformed = []
    for i in range(len(scans)):
        scans_transformed.append(deepcopy(scans[i]))
        scans_transformed[i].transform(traj[i][0], traj[i][1])
    
    map = scans_transformed[0]
    for s in scans_transformed[1:]:
        map = map.merge(s, dist_thresh=dist_thresh)
        map.reduce_inside(p2p_dist_thresh=p2p_thresh)
        map.remove_small_planes(area_thresh=area_thresh)
        map.fuse_edges(vertex_merge_thresh=fuse_thresh)
    
    return map


def offline_slam(PCs, init_pose):
    """Offline SLAM
    
    Process sequence of point clouds to generate a trajectory  
    estimate and map.

    Parameters
    ----------
    PCs : list of np.array 
        List of point clouds

    Returns
    -------
    trajectory : 
        Sequence of poses
    map : Scan
        Final map composed from merged scans
    
    """
    # For airsim
    N = len(PCs)

    # Relative transformations
    R_hats = []
    t_hats = []

    # Absolute poses
    R_abs, t_abs = init_pose
    poses = N * [None]
    poses[0] = (R_abs, t_abs)

    # Scans
    scans = N * [None]
    scans[0] = pc_to_scan(PCs[0])

    # Pose graph

    for i in range(1, N):
        P = PCs[i]
        
        # Extract scan
        scans[i] = pc_to_scan(P)
        scans[i].remove_small_planes(area_thresh=5.0)

        # Registration
        R_hat, t_hat = robust_GN_register(scans[i], scans[i-1])
        t_abs += (R_abs @ t_hat).flatten()
        R_abs = R_hat @ R_abs
        poses[i] = (R_abs, t_abs)



