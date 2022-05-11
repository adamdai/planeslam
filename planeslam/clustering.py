"""Clustering points clouds

"""

import numpy as np


def cluster_mesh_graph_search(mesh, normal_match_thresh=0.17, min_cluster_size=5):
    """Cluster mesh with graph search
    
    Parameters
    ----------
    mesh : LidarMesh
        Mesh object to cluster
    normal_match_thresh : float
        Norm difference threshold to cluster triangles together (default value corresponds to ~10 degrees difference)
    min_cluster_size : int
        Minimum cluster size

    Returns
    -------
    clusters : list of lists
        List of triangle indices grouped into clusters
    avg_normals : list of np.array
        Average normal vectors for each cluster

    """
    # Compute surface normals
    normals = mesh.compute_normals()

    # Graph search
    clusters = []  # Clusters are idxs of triangles, triangles are idxs of points
    avg_normals = []
    to_cluster = set(range(len(mesh.DT.simplices)))

    while to_cluster:
        root = to_cluster.pop()
        avg_normal = normals[root,:]

        cluster = [root]
        search_queue = set(mesh.tri_nbr_dict[root])
        search_queue = set([x for x in search_queue if x in to_cluster])  # Don't search nodes that have already been clustered

        while search_queue:
            i = search_queue.pop()
            if np.linalg.norm(normals[i,:] - avg_normal) < normal_match_thresh:
                # Add node to cluster and remove from to_cluster
                cluster.append(i)
                to_cluster.remove(i)
                # Add its neighbors (that are not already clustered or search queue) to the search queue
                search_nbrs = mesh.tri_nbr_dict[i].copy()
                search_nbrs = [x for x in search_nbrs if x in to_cluster]
                search_nbrs = [x for x in search_nbrs if not x in search_queue]
                search_queue.update(search_nbrs)
                # Update average normal
                avg_normal = np.mean(normals[cluster], axis=0)
                avg_normal = avg_normal / np.linalg.norm(avg_normal)

        if len(cluster) >= min_cluster_size:
            clusters.append(cluster)
            avg_normals.append(avg_normal)

    return clusters, avg_normals


def find_cluster_boundary(cluster, mesh):
    """Find boundary vertices in cluster of triangles

    Parameters
    ----------
    cluster : list
        List of triangle indices denoting cluster
    mesh : LidarMesh
        Mesh which cluster belongs to

    Returns
    -------
    bd_verts : set
        Set containing indices of vertices on boundary of cluster
        
    """
    bd_verts = set()  
    for tri_idx in cluster:
        tri_nbrs = set(mesh.tri_nbr_dict[tri_idx]) & set(cluster)
        if len(tri_nbrs) == 2:
            # 2 vertices not shared by neighbors are boundary points
            nbr_verts = mesh.simplices[list(tri_nbrs),:]
            vals, counts = np.unique(nbr_verts, return_counts=True)
            bd_nbr_verts = set(mesh.simplices[tri_idx,:])
            if 2 in counts:
                bd_nbr_verts.remove(vals[counts==2][0])
            bd_verts.update(bd_nbr_verts)
        elif len(tri_nbrs) == 1:
            # All 3 vertices are boundary points
            bd_verts.update(mesh.simplices[tri_idx])
        
    return bd_verts


def sort_mesh_clusters(clusters, normals=None, reverse=True):
    """Sort mesh clusters of triangles by size

    Parameters
    ----------
    clusters : list of list
        Clusters to sort
    normals : list of np.array, optional
        Associated normals to sort
    reverse : bool, optional
        Whether to sort in reverse order, default True

    Returns
    -------
    clusters : list of list
        Sorted clusters
    normals : list of np.array, optional
        Sorted normals
        
    """
    # Sort clusters from largest to smallest
    if reverse:
        cluster_sort_idx = np.argsort([-len(c) for c in clusters])
    # Sort clusters from smallest to largest
    else:
        cluster_sort_idx = np.argsort([len(c) for c in clusters])
    clusters[:] = [clusters[i] for i in cluster_sort_idx]

    if normals is not None:
        normals[:] = [normals[i] for i in cluster_sort_idx]
        return clusters, normals
    else:
        return clusters

    
def mesh_cluster_pts(mesh, cluster):
    """Get associated points for a mesh triangle cluster

    Parameters
    ----------
    mesh : LidarMesh
        Mesh which cluster belongs to
    cluster : list 
        List of triangle indices forming a cluster

    Returns
    -------
    np.array (n_pts x 3)
        Points in cluster
        
    """
    cluster_pts_idxs = np.unique(mesh.DT.simplices[cluster,:]) 
    return mesh.P[cluster_pts_idxs,:]