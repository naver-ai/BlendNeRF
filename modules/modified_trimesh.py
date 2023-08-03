"""
Refer to https://github.com/mikedh/trimesh
registration.py: Functions for registering (aligning) point clouds with meshes.
"""

import numpy as np

from trimesh import util
from trimesh import bounds
from trimesh import transformations

from trimesh.transformations import transform_points

try:
    from scipy.spatial import cKDTree
except BaseException as E:
    # wrapping just ImportError fails in some cases
    # will raise the error when someone tries to use KDtree
    from trimesh import exceptions
    cKDTree = exceptions.closure(E)

def procrustes(a,
               b,
               reflection=True,
               translation=True,
               scale=True,
               uniform_scale=True,
               rotation=True,
               return_cost=True):
    """
    Perform Procrustes' analysis subject to constraints. Finds the
    transformation T mapping a to b which minimizes the square sum
    distances between Ta and b, also called the cost.

    Parameters
    ----------
    a : (n,3) float
      List of points in space
    b : (n,3) float
      List of points in space
    reflection : bool
      If the transformation is allowed reflections
    translation : bool
      If the transformation is allowed translations
    scale : bool
      If the transformation is allowed scaling
    return_cost : bool
      Whether to return the cost and transformed a as well

    Returns
    ----------
    matrix : (4,4) float
      The transformation matrix sending a to b
    transformed : (n,3) float
      The image of a under the transformation
    cost : float
      The cost of the transformation
    """

    a = np.asanyarray(a, dtype=np.float64)
    b = np.asanyarray(b, dtype=np.float64)
    if not util.is_shape(a, (-1, 3)) or not util.is_shape(b, (-1, 3)):
        raise ValueError('points must be (n,3)!')

    if len(a) != len(b):
        raise ValueError('a and b must contain same number of points!')

    # Remove translation component
    if translation:
        acenter = a.mean(axis=0)
        bcenter = b.mean(axis=0)
    else:
        acenter = np.zeros(a.shape[1])
        bcenter = np.zeros(b.shape[1])

    # Remove scale component
    if scale:
        if uniform_scale:
          ascale = np.sqrt(((a - acenter)**2).sum() / len(a))
          bscale = np.sqrt(((b - bcenter)**2).sum() / len(b))

        else:
          ascale_x = np.sqrt(((a[:, 0] - acenter[0])**2).sum() / len(a))
          ascale_y = np.sqrt(((a[:, 1] - acenter[1])**2).sum() / len(a))
          ascale_z = np.sqrt(((a[:, 2] - acenter[2])**2).sum() / len(a))

          ascale = np.array((ascale_x, ascale_y, ascale_z))

          bscale_x = np.sqrt(((b[:, 0] - bcenter[0])**2).sum() / len(b))
          bscale_y = np.sqrt(((b[:, 1] - bcenter[1])**2).sum() / len(b))
          bscale_z = np.sqrt(((b[:, 2] - bcenter[2])**2).sum() / len(b))

          bscale = np.array((bscale_x, bscale_y, bscale_z))
          

    else:
        ascale = 1
        bscale = 1


    if rotation:
      # Use SVD to find optimal orthogonal matrix R
      # constrained to det(R) = 1 if necessary.
      u, s, vh = np.linalg.svd(
          np.dot(((b - bcenter) / bscale).T, ((a - acenter) / ascale)))

      if reflection:
          R = np.dot(u, vh)
      else:
          # no reflection allowed, so determinant must be 1.0
          R = np.dot(np.dot(u, np.diag(
              [1, 1, np.linalg.det(np.dot(u, vh))])), vh)
    else:
      R = np.identity(3)

    # Compute our 4D transformation matrix encoding
    # a -> (R @ (a - acenter)/ascale) * bscale + bcenter
    #    = (bscale/ascale)R @ a + (bcenter - (bscale/ascale)R @ acenter)
    translation = bcenter - (bscale / ascale) * np.dot(R, acenter)

    matrix = np.hstack(((bscale / ascale) * R, translation.reshape(-1, 1)))
    matrix = np.vstack(
        (matrix, np.array([0.] * (a.shape[1]) + [1.]).reshape(1, -1)))

    if return_cost:
        transformed = transform_points(a, matrix)
        cost = ((b - transformed)**2).mean()
        return matrix, transformed, cost
    else:
        return matrix


def icp(a,
        b,
        initial=np.identity(4),
        threshold=1e-5,
        max_iterations=20,
        **kwargs):
    """
    Apply the iterative closest point algorithm to align a point cloud with
    another point cloud or mesh. Will only produce reasonable results if the
    initial transformation is roughly correct. Initial transformation can be
    found by applying Procrustes' analysis to a suitable set of landmark
    points (often picked manually).

    Parameters
    ----------
    a : (n,3) float
      List of points in space.
    b : (m,3) float or Trimesh
      List of points in space or mesh.
    initial : (4,4) float
      Initial transformation.
    threshold : float
      Stop when change in cost is less than threshold
    max_iterations : int
      Maximum number of iterations
    kwargs : dict
      Args to pass to procrustes

    Returns
    ----------
    matrix : (4,4) float
      The transformation matrix sending a to b
    transformed : (n,3) float
      The image of a under the transformation
    cost : float
      The cost of the transformation
    """

    a = np.asanyarray(a, dtype=np.float64)
    if not util.is_shape(a, (-1, 3)):
        raise ValueError('points must be (n,3)!')

    is_mesh = util.is_instance_named(b, 'Trimesh')
    if not is_mesh:
        b = np.asanyarray(b, dtype=np.float64)
        if not util.is_shape(b, (-1, 3)):
            raise ValueError('points must be (n,3)!')
        btree = cKDTree(b)

    # transform a under initial_transformation
    a = transform_points(a, initial)
    total_matrix = initial

    # start with infinite cost
    old_cost = np.inf

    # avoid looping forever by capping iterations
    for n_iteration in range(max_iterations):
        # Closest point in b to each point in a
        if is_mesh:
            closest, distance, faces = b.nearest.on_surface(a)
        else:
            distances, ix = btree.query(a, 1)
            closest = b[ix]

        # align a with closest points
        matrix, transformed, cost = procrustes(a=a,
                                               b=closest,
                                               **kwargs)

        # update a with our new transformed points
        a = transformed
        total_matrix = np.dot(matrix, total_matrix)

        if old_cost - cost < threshold:
            break
        else:
            old_cost = cost

    return total_matrix, transformed, cost
