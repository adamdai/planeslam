"""RRT

This module defines functions for RRT generation.

"""

import numpy as np
import time 
import plotly.graph_objects as go

import planeslam.planning.params as params
from planeslam.planning.LPM import LPM
import planeslam.planning.utils as utils


class RRT:
    """RRT class

    Attributes
    ----------

    Methods
    -------

    """

    def __init__(self, map, root, radius=5, nodes=1000):
        """Constructor

        Parameters
        ----------
        map : Scan
            Map.
        root : np.array (3)
            Root node position.

        """
        self.root = root

        self.G_pos = root
        self.G_edges = []

        self.radius = radius

        self.map = map
        
        # Compute world bounds from map
        self.bounds = map.compute_bounds()

        self.node_ct = 1

        while self.node_ct < nodes:
            self.add_node()



    def add_node(self):
        """Add new node
        
        """
        # Sample a random position in bounds
        rand_sample = np.random.random_sample(size=3)
        rand_pos = self.bounds[:,0] + rand_sample * np.diff(self.bounds).flatten()

        # Find nearest node in G
        if self.G_pos.ndim > 1:
            dists = np.linalg.norm(self.G_pos - rand_pos, axis=1)
            nearest = np.argmin(dists)
            nearest_pos = self.G_pos[nearest]
            dist_nearest = dists[nearest]
        else:
            nearest = 0
            nearest_pos = self.G_pos
            dist_nearest = np.linalg.norm(self.G_pos - rand_pos)

        # Find new vertex along path and check for collision
        new_pos = nearest_pos + (self.radius / dist_nearest) * (rand_pos - nearest_pos)
        line = np.vstack((nearest_pos, new_pos))
        collision = False
        
        # Collision-check
        for plane in self.get_nearby_planes(nearest_pos):
            if plane.check_line_intersect(line):
                collision = True
        
        # If no collision, add new vertex to G
        if not collision:
            self.G_pos = np.vstack((self.G_pos, new_pos))
            self.G_edges.append((nearest, self.node_ct))
            self.node_ct += 1

    
    def get_nearby_planes(self, pos):
        """Find planes in map which are within radius of position
        
        """
        nearby_planes = []
        for plane in self.map.planes:
            if plane.dist_to_point(pos) < self.radius:
                nearby_planes.append(plane)
        return nearby_planes


    def plot_trace(self, line_width=5, marker_size=5):
        """Generate plot trace for plotly visualization

        Returns
        -------
        list
            List of plotly.go traces for RRT visualization
        
        """
        # Visualize
        edge_x = []
        edge_y = []
        edge_z = []
        for edge in self.G_edges:
            x0, y0, z0 = self.G_pos[edge[0]]
            x1, y1, z1 = self.G_pos[edge[1]]
            edge_x.append(x0)
            edge_x.append(x1)
            edge_x.append(None)
            edge_y.append(y0)
            edge_y.append(y1)
            edge_y.append(None)
            edge_z.append(z0)
            edge_z.append(z1)
            edge_z.append(None)

        edge_trace = go.Scatter3d(x=edge_x, y=edge_y, z=edge_z, mode='lines', line=dict(width=line_width, color='orange'))
        node_trace = go.Scatter3d(x=self.G_pos[:,0], y=self.G_pos[:,1], z=self.G_pos[:,2], mode='markers', marker=dict(size=marker_size, color='coral'))

        return [edge_trace, node_trace]


        
