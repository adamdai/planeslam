"""Test whole SLAM pipeline 

"""

import numpy as np
import os
import time
import plotly.graph_objects as go
from copy import deepcopy

import planeslam.io as io
from planeslam.scan import pc_to_scan
from planeslam.general import NED_to_ENU
from planeslam.geometry.util import quat_to_R
from planeslam.pose_graph import PoseGraph
from planeslam.registration import robust_GN_register, loop_closure_register
from planeslam.slam import generate_map


if __name__ == "__main__":

    # Read in point cloud data
    print("reading data...")
    binpath = os.path.join(os.getcwd(), '..', 'data', 'airsim', 'blocks_loop_225_samples_3hz_spd_4_noyaw', 'lidar', 'Drone0')
    PCs = io.read_lidar_bin(binpath)

    # Read in ground-truth poses (in drone local frame)
    posepath = os.path.join(os.getcwd(), '..', 'data', 'airsim', 'blocks_loop_225_samples_3hz_spd_4_noyaw', 'poses', 'Drone0')
    drone_positions, drone_orientations = io.read_poses(posepath)

    # Convert to ENU
    num_scans = len(PCs)

    for i in range(num_scans):
        PCs[i] = NED_to_ENU(PCs[i])

    drone_positions = NED_to_ENU(drone_positions)
    drone_orientations = NED_to_ENU(drone_orientations)

    drone_rotations = np.zeros((3,3,num_scans))
    for i in range(num_scans):
        drone_rotations[:,:,i] = quat_to_R(drone_orientations[i])

    # Run SLAM
    print("running SLAM...")
    LOOP_CLOSURE_SEARCH_RADIUS = 10  # [m]
    LOOP_CLOSURE_PREV_THRESH = 50  # don't search for loop closures over this number of the previous scans
    init_pose = (quat_to_R(drone_orientations[0]), drone_positions[0,:].copy())

    LC_TIMEOUT = 10
    LC_timer = -1

    #--------------------------------------------------------------#
    N = len(PCs)

    # Relative transformations
    R_hats = []
    t_hats = []

    # Absolute poses
    R_abs, t_abs = init_pose
    poses = [(R_abs, t_abs)]
    positions = t_abs

    # Scans
    scans = [pc_to_scan(PCs[0])]
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
        marker=dict(size=3, color='orange'), line=dict(width=4), showlegend=False)
    fig = go.Figure(data=map.plot_trace(colors=['blue'], showlegend=False)+[traj_trace])
    fig.update_layout(scene=dict(aspectmode='data', 
        xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)))
    imgpath = os.path.join(os.getcwd(), '..', 'images', 'airsim', 'run_1', 'map_0.png')
    fig.write_image(imgpath, width=2500, height=1600)

    for i in range(1, N):
        print("i = ", i)
        P = PCs[i]
        
        # Extract scan
        start_time = time.time()
        scans.append(pc_to_scan(P))
        scans[i].remove_small_planes(area_thresh=5.0)
        extraction_times.append(time.time() - start_time)
        
        # Registration
        start_time = time.time()
        R_hat, t_hat = robust_GN_register(scans[i], scans[i-1])
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
        if LC_timer < 0:
            if i > LOOP_CLOSURE_PREV_THRESH:
                LC_dists = np.linalg.norm(t_abs - positions[:i-LOOP_CLOSURE_PREV_THRESH], axis=1)
                LCs = np.argwhere(LC_dists < LOOP_CLOSURE_SEARCH_RADIUS)
                if len(LCs) > 0:
                    # Find the lowest distance loop closure
                    j = LCs[np.argsort(LC_dists[LCs].flatten())[0]][0]
                    print(f'adding loop closure: ({i}, {j})')
                    R_LC, t_LC = loop_closure_register(scans[i], scans[j], poses[i], poses[j], t_loss_thresh=0.1)
                    # Add LC edge
                    g.add_edge([j, i], (R_LC, t_LC))
                    # Optimize graph
                    g.optimize()    
                    # TODO: Re-create map
                    map = generate_map(g.get_poses(), scans)
                    LC = True
                    LC_timer = LC_TIMEOUT

                    R_abs, t_abs = g.get_poses()[-1]
        else:
            LC_timer -= 1
        loop_closure_times.append(time.time() - start_time)

        # Map update (merging)
        start_time = time.time()
        if not LC:
            map = map.merge(scan_transformed, dist_thresh=7.5)
            map.reduce_inside(p2p_dist_thresh=5)
            map.fuse_edges(vertex_merge_thresh=2.0)
        merging_times.append(time.time() - start_time)

        # Visualization
        positions = g.get_positions()
        traj_trace = go.Scatter3d(x=positions[:,0], y=positions[:,1], z=positions[:,2], 
            marker=dict(size=3, color='orange'), line=dict(width=4), showlegend=False)
        fig = go.Figure(data=map.plot_trace(colors=['blue'], showlegend=False)+[traj_trace])
        fig.update_layout(scene=dict(aspectmode='data', 
            xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)))
        imgpath = os.path.join(os.getcwd(), '..', 'images', 'airsim', 'run_1', 'map_'+str(i)+'.png')
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