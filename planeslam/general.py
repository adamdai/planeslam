"""General utilities

"""

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px


def NED_to_ENU(P):
    """Convert a set of 3D points from NED to ENU

    Parameters
    ----------
    P : np.array (n_pts x 3)
        NED points to convert

    Returns
    -------
    np.array (n_pts x 3)
        points in ENU

    """
    P[:,[0,1]] = P[:,[1,0]]  # Swap x and y
    P[:,2] = -P[:,2]  # Negate z
    return P


def remap(arr, k, v):
    """Remap the elements of an array using the map k->v

    https://stackoverflow.com/questions/55949809/efficiently-replace-elements-in-array-based-on-dictionary-numpy-python

    Elements of arr which do not appear in k will be unchanged. Works for 
    k and v non-negative integers.

    Parameters
    ----------
    arr : np.array 
        Array to remap
    k : np.array (1 x n_vals)
        Values to replace (i.e. "keys")
    v : np.array (1 x n_vals)
        Values to replace with (i.e. "values")

    Returns
    -------
    np.array 
        Remapped array

    """
    assert len(k) == len(v), "k and v should have same length"
    mapping_ar = np.arange(arr.max()+1)
    mapping_ar[k] = v
    return mapping_ar[arr]
    # map = dict(zip(k,v))
    # return np.vectorize(map.get)(arr)


def downsample(arr, factor, axis):
    """Downsample a (2D) numpy array

    Parameters
    ----------
    arr : np.array
        Array to downsample
    factor : int 
        Factor to downsample by
    axis : int (0 or 1)
        Axis to downsample along

    Returns
    -------
    np.array 
        Downsampled array

    """
    assert axis == 0 or axis == 1, "axis should be 0 or 1"
    if axis == 0:
        return arr[::factor,:]
    elif axis == 1:
        return arr[:,::factor]


def normalize(v):
    """Normalize numpy vector

    Parameters
    ----------
    v : np.array 
        Vector to normalize

    Returns
    -------
    np.array 
        Normalized vector

    """
    return v / np.linalg.norm(v)


def plot_3D_setup(P=None, figsize=(15,10)):
    """Setup matplotlib 3D plotting

    Parameters
    ----------
    P : np.array (n_pts x 3)
        Points to use for axis scaling
    
    Returns
    -------
    ax : axes
        Axes for plotting
    
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1, projection='3d')

    if P is not None:
        ax.set_box_aspect((np.ptp(P[:,0]), np.ptp(P[:,1]), np.ptp(P[:,2])))
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    return ax


def color_legend(ax, num_colors):
    """Display legend for matplotlib default colors
    
    Parameters
    ----------
    ax : axes
        Axes to display on
    num_colors : int
        Number of colors in legend

    """
    for i in range(num_colors):
        ax.plot(0, 0, 0, color='C'+str(i), label=str(i))
    ax.legend()


def pc_plot_trace(P):
    """Generate plotly plot trace for point cloud

    Returns
    -------
    fig : plotly go.Scatter3d
        Scatter plot trace

    """
    return go.Scatter3d(x=P[:,0], y=P[:,1], z=P[:,2], 
        mode='markers', marker=dict(size=2))
    

def plot_normals(P,normals,scale=10.):
    """Plot point cloud using plotly

    Returns
    -------
    fig : plotly go.Figure
        Figure handle

    """
    n = P.shape[0]
    points = go.Scatter3d(x=P[:,0], y=P[:,1], z=P[:,2], 
        mode='markers', marker=dict(size=2))
    data = [points]
    xs = [None for i in range(3*n)]
    ys = [None for i in range(3*n)]
    zs = [None for i in range(3*n)]
    for i in range(n):
        xs[3*i] = P[i,0]
        xs[3*i+1] = P[i,0]+scale*normals[i,0]
        ys[3*i] = P[i,1]
        ys[3*i+1] = P[i,1]+scale*normals[i,1]
        zs[3*i] = P[i,2]
        zs[3*i+1] = P[i,2]+scale*normals[i,2]

    data.append(go.Scatter3d(x=xs,y=ys,z=zs,mode="lines"))

    fig = go.Figure(data=data)
    fig.update_layout(width=1000, height=600, scene=dict(
                    aspectmode='data'),showlegend=False)
    return fig
