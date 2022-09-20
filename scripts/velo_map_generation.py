"""Test whole SLAM pipeline 

"""

import numpy as np
import os
import time
import plotly.graph_objects as go
from copy import deepcopy

import planeslam.io as io
from planeslam.scan import velo_pc_to_scan
from planeslam.general import NED_to_ENU
from planeslam.geometry.util import quat_to_R
from planeslam.pose_graph import PoseGraph
from planeslam.registration import robust_GN_register, loop_closure_register
from planeslam.slam import generate_map
from planeslam.point_cloud import velo_preprocess


if __name__ == "__main__":

    # Read in point cloud data
    print("reading data...")
    pcpath = os.path.join(os.getcwd(), '..', 'data', 'velodyne', '9_19_2022', 'flightroom', 'run_3', 'pcs')
    PCs = []
    select_idxs = np.arange(0, len(os.listdir(pcpath)), 5)
    for i in select_idxs:  
        filename = os.path.join(pcpath, 'pc_'+str(i)+'.npy')
        PC = np.load(filename)
        PCs.append(PC)

    # Read in ground-truth poses (in drone local frame)
    posepath = os.path.join(os.getcwd(), '..', 'data', 'velodyne', '9_19_2022', 'flightroom', 'run_3', 'poses')
    poses = []
    for i in select_idxs:  
        filename = os.path.join(posepath, 'pose_'+str(i)+'.npy')
        pose = np.load(filename)
        poses.append(pose)

    rover_rotations = np.zeros((3,3,len(poses)))
    for i in range(len(poses)):
        rover_rotations[:,:,i] = quat_to_R(poses[i][3:])

    # Transform ground truth to account for LiDAR offset
    offset_vec = np.array([0.09, 0.0, 0.0])
    shifted_poses = deepcopy(poses)

    for i in range(len(PCs)):
        # Rotate offset by current pose
        shifted_poses[i][:3] += rover_rotations[:,:,i] @ offset_vec

    rover_positions = np.asarray(poses)[:,:3]
    rover_positions_shifted = np.asarray(shifted_poses)[:,:3]

    # Run SLAM
    print("running SLAM...")
    LOOP_CLOSURE_SEARCH_RADIUS = 0.2  # [m]
    LOOP_CLOSURE_PREV_THRESH = 50  # don't search for loop closures over this number of the previous scans

    start = 0
    init_pose = (rover_rotations[:,:,start], rover_positions_shifted[start,:].copy())

    #--------------------------------------------------------------#
    N = len(PCs)
    #N = 5

    # Relative transformations
    R_hats = []
    t_hats = []

    # Absolute poses
    R_abs, t_abs = init_pose
    poses = [(R_abs, t_abs)]
    positions = t_abs

    # Scans
    scans = [velo_pc_to_scan(velo_preprocess(PCs[start], shifted_poses[start]))]
    scan_transformed = deepcopy(scans[0])
    scan_transformed.transform(R_abs, t_abs)

    # Initalize pose graph
    g = PoseGraph()
    g.add_vertex(0, poses[0])

    # Initialize map
    map = scan_transformed

    #avg_runtime = 0
    extraction_times = []
    registration_times = []
    loop_closure_times = []
    merging_times = []

    camera = dict(
        up=dict(x=0, y=0, z=1),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=0, y=-2.0, z=1.0)
    )

    positions = g.get_positions()
    traj_trace = go.Scatter3d(x=positions[:,0], y=positions[:,1], z=positions[:,2], 
        marker=dict(size=5, color='orange'), line=dict(width=2), showlegend=False)
    fig = go.Figure(data=map.plot_trace(colors=['blue'], showlegend=False)+[traj_trace])
    fig.update_layout(scene=dict(aspectmode='data', 
        xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)))
    imgpath = os.path.join(os.getcwd(), '..', 'images', 'map', 'rover', 'run_3', 'map_0.png')
    fig.write_image(imgpath, width=2500, height=1600)

    for i in range(1, N-start):
        print("i = ", i)
        P = velo_preprocess(PCs[i+start], shifted_poses[i+start])
        
        # Extract scan
        start_time = time.time()
        scans.append(velo_pc_to_scan(P))
        scans[i].remove_small_planes(area_thresh=0.1)
        scans[i].reduce_inside(p2p_dist_thresh=0.1)
        extraction_times.append(time.time() - start_time)
        
        # Registration
        start_time = time.time()
        R_hat, t_hat = robust_GN_register(scans[i], scans[i-1], c2c_thresh=1.0)
        registration_times.append(time.time() - start_time)
        t_abs += (R_abs @ t_hat).flatten()
        R_abs = R_hat @ R_abs

        # Transform scan
        scan_transformed = deepcopy(scans[i])
        scan_transformed.transform(R_abs, t_abs)

        # Save data
        R_hats.append(R_hat)
        t_hats.append(t_hat)
        positions = np.vstack((positions, t_abs))
        poses.append((R_abs.copy(), t_abs.copy()))

        # Pose graph update
        g.add_vertex(i, poses[i])
        g.add_edge([i-1, i], (R_hat, t_hat))

        # Loop closure detection
        LC = False
        if i > LOOP_CLOSURE_PREV_THRESH:
            LC_dists = np.linalg.norm(t_abs - positions[:i-LOOP_CLOSURE_PREV_THRESH], axis=1)
            LCs = np.argwhere(LC_dists < LOOP_CLOSURE_SEARCH_RADIUS)
            if len(LCs) > 0:
                # Find the lowest distance loop closure
                j = LCs[np.argsort(LC_dists[LCs].flatten())[0]][0]
                #print(f'adding loop closure: ({i}, {j})')
                R_LC, t_LC = loop_closure_register(scans[i], scans[j], poses[i], poses[j], t_loss_thresh=0.1)
                # Add LC edge
                g.add_edge([j, i], (R_LC, t_LC))
                # Optimize graph
                g.optimize()    
                # Re-create map
                map = generate_map(g.get_poses(), scans, dist_thresh=0.1, p2p_thresh=0.1, area_thresh=0.1, fuse_thresh=0.1)
                LC = True
        loop_closure_times.append(time.time() - start_time)

        # Map update (merging)
        start_time = time.time()
        if not LC:
            map = map.merge(scan_transformed, dist_thresh=0.1)
            map.reduce_inside(p2p_dist_thresh=0.1)
            map.fuse_edges(vertex_merge_thresh=0.1)
        merging_times.append(time.time() - start_time)

        # Visualization
        positions = g.get_positions()
        traj_trace = go.Scatter3d(x=positions[:,0], y=positions[:,1], z=positions[:,2], 
            marker=dict(size=5, color='orange'), showlegend=False)
        fig = go.Figure(data=map.plot_trace(colors=['blue'], showlegend=False)+[traj_trace])
        fig.update_layout(scene=dict(aspectmode='data', 
            xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)))
        imgpath = os.path.join(os.getcwd(), '..', 'images', 'map', 'rover', 'run_3', 'map_'+str(i)+'.png')
        fig.write_image(imgpath, width=2500, height=1600)

        #avg_runtime += time.time() - start_time

    #avg_runtime /= N-1
    #print("Done. Avg runtime: ", avg_runtime)

    print(f"Averages: \n \
            extraction: {np.mean(extraction_times)} \n \
            registration: {np.mean(registration_times)} \n \
            loop closure: {np.mean(loop_closure_times)} \n \
            merging: {np.mean(merging_times)} \n \
            total: {np.mean(extraction_times) + np.mean(registration_times) + np.mean(loop_closure_times) + np.mean(merging_times)}")

    print(f"STD: \n \
            extraction: {np.std(extraction_times)} \n \
            registration: {np.std(registration_times)} \n \
            loop closure: {np.std(loop_closure_times)} \n \
            merging: {np.std(merging_times)} \n \
            total: {np.sqrt(np.mean([np.var(extraction_times), np.var(registration_times), np.var(loop_closure_times), np.var(merging_times)]))}")