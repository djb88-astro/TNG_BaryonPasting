import numpy as np
from numba import njit, jitclass, float64, boolean, optional, deferred_type

node_type = deferred_type()

spec = [('p', float64[:,:]), ('m', float64[:]), ('s', float64[:]), ('Mtot', float64),
        ('Smax', float64), ('bbox', float64[:,:]), ('extent', float64), ('Leaf', boolean),
        ('CoM', float64[:]), ('delta', float64), ('Left', boolean), ('Right', boolean),
        ('Cleft', optional(node_type)), ('Cright', optional(node_type))]

"""
Functions here build a KD of particle positions and computes the
gravitional potential of the halo from it.
"""

"""
Class creates a KD-tree from the particle distribtion. Heavy use
of Numba to try and make it computationally faster, however this
is still very expensive.

Arguments:
  -points    : 2D ARRAY of particle positions
  -masses    : ARRAY of particle weights, typically their mass
  -softening : Value of the gravitational force softening length [FLOAT]
"""

@jitclass(spec)
class Node(object):
    def __init__(self, points, masses, softening):
        # Store inputs, compute basics
        self.p    = points
        self.m    = masses
        self.s    = softening
        self.Mtot = self.m.sum()
        self.Smax = self.s.max()

        # Domain extent
        self.bbox = np.array([[self.p[:,0].min(), self.p[:,0].max()],
                              [self.p[:,1].min(), self.p[:,1].max()],
                              [self.p[:,2].min(), self.p[:,2].max()]])
        self.extent = (self.bbox[:,1] - self.bbox[:,0]).max()

        # Determine if a leaf of the tree
        if len(self.p) == 1:
            self.Leaf = True
            self.CoM  = self.p[0]
        else:
            self.Leaf = False
            self.CoM  = np.zeros(3)
            self.CoM[0] = (self.p[:,0] * self.m).sum() / self.Mtot
            self.CoM[1] = (self.p[:,1] * self.m).sum() / self.Mtot
            self.CoM[1] = (self.p[:,2] * self.m).sum() / self.Mtot

            self.delta  = 0.0
            self.delta += (0.5 * self.bbox[0].sum() - self.CoM[0]) ** 2.0
            self.delta += (0.5 * self.bbox[1].sum() - self.CoM[1]) ** 2.0
            self.delta += (0.5 * self.bbox[2].sum() - self.CoM[2]) ** 2.0
            self.delta  = np.sqrt(self.delta)

        # Arguments for nested Child classes of this Node
        self.Left   = False
        self.Right  = False
        self.Cleft  = None
        self.Cright = None
        return

    def Generate_Children(self, axis):
        """
        Generate children of current branch

        Arguments:
          -axis : Current Euclidean axis being considered
        """

        if self.Leaf: return False
        # If not a Leaf, compute midpoint and particles left/right of it
        x   = self.p[:,axis]
        med = 0.5 * self.bbox[axis].sum()
        idx = (x < med)
        # Check for particles left of midpoint, create new Node instance if required
        if np.any(idx):
            self.Cleft = Node(self.p[idx], self.m[idx], self.s[idx])
            self.Left  = True
        # Now reverse indices, repeat for right
        idx = np.invert(idx)
        if np.any(idx):
            self.Cright = Node(self.p[idx], self.m[idx], self.s[idx])
            self.Right  = True
        # Finish this Node and return
        self.p = np.empty((1,1))
        self.m = np.empty(1)
        self.s = np.empty(1)
        return True

node_type.define(Node.class_type.instance_type)

@njit
def construct_tree(p, m, s):
    """
    Build a KD tree

    Arguments:
      -p : Position
      -m : Weight
      -s : Node width

    Returns:
      -Root : KD tree class instance
    """

    Root = Node(p, m, s)

    nodes     = [Root]
    axis      = 0
    divisible = True
    Ctotal    = 0
    while divisible:
        N          = len(nodes)
        divisible  = False
        count      = 0
        for j in np.arange(N)[Ctotal:]:
            if nodes[j].Leaf: continue

            divisible = nodes[j].Generate_Children(axis)
            if nodes[j].Left:
                nodes.append(nodes[j].Cleft)
                count += 1
            if nodes[j].Right:
                nodes.append(nodes[j].Cright)
                count += 1

        axis = (axis + 1) % 3
        if divisible:
            Ctotal = len(nodes) - count
    return Root

@njit
def compute_potential_via_tree(p, tree, G=6.67430e-8, theta=1.0):
    """
    Compute the potential of each particle -- assumes CGS units

    Arguments:
      -p     : Position [ARRAY]
      -tree  : Instance of the KD tree class 
      -G     : Newton's gravitational constant in CGS units [FLOAT]
      -theta : Opening angle, determines the force accuracy [FLOAT]

    Returns:
      -pot : Gravitational potential at this positions
    """

    pot = np.zeros(len(p))
    for j in range(len(p)):
        pot[j] = G * tree_walk(p[j], tree, 0.0, theta=theta)
    return pot

@njit(fastmath=True)
def tree_walk(pos, tree, phi, theta=1.0):
    """                                                                         
    Does the hard work of walking the tree and computing the potential

    Arguments:
      -pos   : 2D ARRAY of positions
      -tree  : Instance of the KD tree class
      -phi   : Gravitional potential [FLOAT]
      -theta : Opening angle, determines the force accuracy [FLOAT]

    Returns:
      -phi : Gravitional potential [FLOAT]
    """

    # Distance to point
    dx = tree.CoM - pos
    r  = np.sqrt(dx[0] * dx[0] + dx[1] * dx[1] + dx[2] * dx[2])

    # Contribution to potential
    if tree.Leaf:
        if r > 0: phi += tree.Mtot * kernel(r, tree.Smax)
    elif r > max(tree.extent / theta, tree.Smax + tree.extent):
        phi -= tree.Mtot / r
    else:
        if tree.Left:
            phi = tree_walk(pos, tree.Cleft, phi, theta=theta)
        if tree.Right:
            phi = tree_walk(pos, tree.Cright, phi, theta=theta)
    return phi

@njit(fastmath=True)
def kernel(r, h):
    """                                                                         
    Compute the cubic spline softened kernel

    Arguments:
      -r : Radial distance [FLOAT]
      -h : Gravitational softening length [FLOAT]

    Returns:
      Kernel value
    """

    if h == 0.0: return -1.0 / r

    hinv = 1.0 / h
    q    = r * hinv
    if q <= 0.5:
         return (-2.8 + q * q * (5.33333333333333333 + q * q * (6.4 * q - 9.6))) * hinv
    elif q <= 1:
        return (-3.2 + 0.066666666666666666666 / q + q * q * (10.666666666666666666666 + q * (-16.0 + q * (9.6 - 2.1333333333333333333333 * q)))) * hinv
    else:
        return -1./r
    return
