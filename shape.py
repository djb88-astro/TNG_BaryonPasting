import numpy as np

def iterative_cumulative_shape_measure(pos, weights, rmin=0.0, rmax=1.0, ITER_MAX=10, TOL=0.01):
    """
    Measure the shape of a halo within a given aperture

    Arguments:
      -pos      : ARRAY containing 3D particle positions
      -weights  : ARRAY containing particle weight (typically mass)
      -rmin     : Minimum extent of aperture [FLOAT]
      -rmax     : Maximum extent of aperture [FLOAT] 
      -ITER_MAX : INTEGER specifying the maximum number of iterations
      -TOL      : Tolerence that defines convergence [FLOAT]

    Returns:
      -Q  : Ratio of semi-major to major axes [FLOAT]
      -S  : Ratio of minor to major axes [FLOAT]
      -Iv : Eigenvectors of final shape [2D ARRAY]
    """

    # Perform intial inertial tensor and shape calculation
    r   = np.sqrt((pos ** 2.0).sum(axis=-1))
    rdx = np.where((r > rmin) & (r <= rmax))[0]
    p   = np.copy(pos)
    w   = np.copy(weights)

    Iten              = compute_inertial_tensor(p[rdx], w[rdx])
    Ivalues, Ivectors = compute_eigenvalues_and_vectors(Iten)
    q                 = Ivalues[1] / Ivalues[0]
    s                 = Ivalues[2] / Ivalues[0]
    
    # Now iterate
    for j in range(0, ITER_MAX, 1):
        # Rotate into frame
        RM    = Ivectors.T
        p_rot = rotate_vectors_cm(RM, p)
        p     = np.copy(p_rot)

        # Reselect those still within aperture
        r   = np.sqrt(p_rot[:,0] ** 2.0 + (p_rot[:,1] / q) ** 2.0 + (p_rot[:,2] / s) ** 2.0)
        rdx = np.where((r > rmin) & (r <= rmax))[0]

        # New inertial tensor, shape calc.
        Iten              = compute_inertial_tensor(p_rot[rdx], w[rdx])
        Ivalues, Ivectors = compute_eigenvalues_and_vectors(Iten)

        # Compare updated shape values, break if converged
        q_new = Ivalues[1] / Ivalues[0]
        s_new = Ivalues[2] / Ivalues[0]
        if abs((q_new - q) / q) < TOL and abs((s_new - s) / s) < TOL:
            q = q_new
            s = s_new
            break
        else:
            q = q_new
            s = s_new
    
    return q, s, Ivectors

def iterative_radial_shape_profile(pos, weights, R200, rmin=0.05, rmax=5.0, Nb=25, ITER_MAX=10, \
                                   TOL=0.01, IBzero=True):
    """
    Measure halo shape in radial annuli

    Arguments:
      -pos      : ARRAY containing 3D particle positions
      -weights  : ARRAY containing particle weight (typically mass)
      -R200     : Halo aperture to normalize radial bins [FLOAT]
      -rmin     : Minimum extent of radial profile [FLOAT]
      -rmax     : Maximum extent of radial profile [FLOAT]
      -Nb       : Number of radial bins [INTEGER]
      -ITER_MAX : INTEGER specifying the maximum number of iterations
      -TOL      : Tolerence that defines convergence [FLOAT]
      -IBzero   : BOOLEAN, if TRUE reset inner most bin edge to zero

    Returns:
      -Q  : Ratio of semi-major to major axes [ARRAY]
      -S  : Ratio of minor to major axes [ARRAY]
      -Iv : Eigenvectors of final shapes [3D ARRAY]
    """

    # Set up radial bins -- zero inner most edge
    bins = np.logspace(np.log10(rmin), np.log10(rmax), num=Nb + 1) * R200
    cens = 10.0 ** (0.5 * (np.log10(bins[1:]) + np.log10(bins[:-1])))
    if IBzero: bins[0] = 0.0

    # Perform intial inertial tensor and shape calculation - copy here is overkill
    r   = np.sqrt((pos ** 2.0).sum(axis=-1))
    rdx = np.digitize(r, bins) - 1
    idx = np.where((rdx >= 0) & (rdx < Nb))[0]
    p   = np.copy(pos)
    w   = np.copy(weights)

    Iten              = compute_inertial_tensor_radial_bins(Nb, p[idx], w[idx], rdx[idx])
    Ivalues, Ivectors = compute_eigenvalues_and_vectors_radial(Iten)
    q                 = Ivalues[:,1] / Ivalues[:,0]
    s                 = Ivalues[:,2] / Ivalues[:,0]

    # Now iterate
    for j in range(0, ITER_MAX, 1):
        # Rotate into frame
        RM         = Ivectors.transpose(0,2,1)
        p_rot      = np.copy(p)
        p_rot[idx] = rotate_vectors_rp(rdx[idx], RM, p[idx])
        p          = np.copy(p_rot)
        
        # Reselect those still within aperture
        for k in np.unique(rdx[idx]):
            r[rdx == k] = np.sqrt(p_rot[rdx == k,0] ** 2.0 \
                                  + (p_rot[rdx == k,1] / q[k]) ** 2.0 \
                                  + (p_rot[rdx == k,2] / s[k]) ** 2.0)
        rdx = np.digitize(r, bins) - 1
        idx = np.where((rdx >= 0) & (rdx < Nb))[0]

        # New inertial tensor, eigenvalues
        Iten              = compute_inertial_tensor_radial_bins(Nb, p_rot[idx], w[idx], rdx[idx])
        Ivalues, Ivectors = compute_eigenvalues_and_vectors_radial(Iten)

        # New shape, check for convergence
        q_new = Ivalues[:,1] / Ivalues[:,0]
        s_new = Ivalues[:,2] / Ivalues[:,0]        
        if (np.abs((q_new - q) / q)).max() < TOL and (np.abs((s_new - s) / s)).max() < TOL:
            q = q_new
            s = s_new
            break

        q = q_new
        s = s_new
    return q, s, Ivectors

def compute_inertial_tensor(pos, weights):
    """
    Compute the inertial tensor of the particle distribution

    Arguments:
      -pos     : ARRAY containing 3D particle positions
      -weights : ARRAY containing particle weight

    Returns:
      -Iten : Inertial tensor
    """

    Iten      = np.zeros((3, 3), dtype=np.float)
    Iten[0,0] = np.sum(weights * pos[:,0] * pos[:,0])
    Iten[1,1] = np.sum(weights * pos[:,1] * pos[:,1])
    Iten[2,2] = np.sum(weights * pos[:,2] * pos[:,2])
    Iten[0,1] = np.sum(weights * pos[:,0] * pos[:,1])
    Iten[1,0] = Iten[0,1]
    Iten[0,2] = np.sum(weights * pos[:,0] * pos[:,2])
    Iten[2,0] = Iten[0,2]
    Iten[1,2] = np.sum(weights * pos[:,1] * pos[:,2])
    Iten[2,1] = Iten[1,2]
    return Iten / np.sum(weights)

def compute_inertial_tensor_radial_bins(Nb, pos, weights, idx):
    """
    Compute the inertial tensors of radial annuli

    Arguments:
      -Nb      : Number of radial annuli [INTEGER]
      -pos     : ARRAY containing 3D particle positions
      -weights : ARRAY containing particle weight
      -idx     : ARRAY specifying which radial bin the particle occupies

    Returns:
      -Iten : ARRAY of inertial tensors
    """

    # Total weight in each annuli
    Wsum = np.zeros(Nb, dtype=np.float)
    np.add.at(Wsum, idx, weights)

    # Inertial tensors
    Iten = np.zeros((Nb, 3, 3), dtype=np.float)
    np.add.at(Iten[:,0,0], idx, weights * pos[:,0] * pos[:,0])
    np.add.at(Iten[:,1,1], idx, weights * pos[:,1] * pos[:,1])
    np.add.at(Iten[:,2,2], idx, weights * pos[:,2] * pos[:,2])
    np.add.at(Iten[:,0,1], idx, weights * pos[:,0] * pos[:,1])
    np.add.at(Iten[:,0,2], idx, weights * pos[:,0] * pos[:,2])
    np.add.at(Iten[:,1,2], idx, weights * pos[:,1] * pos[:,2])
    Iten[:,1,0] = Iten[:,0,1]
    Iten[:,2,0] = Iten[:,0,2]
    Iten[:,2,1] = Iten[:,1,2]

    for j in np.unique(idx): Iten[j] /= Wsum[j]
    return Iten

def compute_eigenvalues_and_vectors(Iten):
    """
    Compute the eigenvalues and eigenvectors of an inertial tensor. Return
    them ordered by values, largest to smallest.
    NOTE: We take the square root of the values here as we really want the
    axes lengths.

    Arguments:
      -Iten : Inertial tensor [2D ARRAY]

    Returns:
      -values  : ARRAY of axes lengths (square to get back to eigenvalues)
      -vectors : 2D ARRAY of eigenvectors 
    """

    values, vectors = np.linalg.eigh(Iten)

    values  = np.sqrt(values[::-1])
    vectors = vectors[:,::-1]
    return values, vectors

def compute_eigenvalues_and_vectors_radial(Iten):
    """
    Compute the eigenvalues and eigenvectors of a series of inertial tensors.
    Return them ordered by values, largest to smallest for each annulus.
    NOTE: We take the square root of the values here as we really want the
    axes lengths.

    Arguments:
      -Iten : Inertial tensors as a function radius [3D ARRAY]

    Returns:
      -values  : 2D ARRAY of axes lengths (square to get back to eigenvalues)
      -vectors : 3D ARRAY of eigenvectors
    """

    # Compute values and vectors
    values, vectors = np.linalg.eigh(Iten)

    # Check and correct for negative values due to machine precision issues
    values[values < 0.0] = 1.0e-10

    # Now sorted values and vectors into correct order and return
    values  = np.sqrt(values[:,::-1])
    vectors = vectors[:,:,::-1]
    return values, vectors

def rotate_vectors_cm(RM, pos):
    """
    Rotate vectors by a rotation matrix
    """

    return np.dot(RM, pos.T).T

def rotate_vectors_rp(idx, RM, pos):
    """
    Rotate a series of vectors by a series of rotation matrices
    """

    tmp = np.zeros(pos.shape, dtype=np.float)
    for k in np.unique(idx): tmp[idx == k] = np.dot(RM[k], pos[idx == k].T).T
    return tmp

def rotate_vectors_Iv(Nb, RM, pos):
    """
    Rotate a series of eigenvectors by a series of rotation matrices
    """

    for k in np.arange(Nb): pos[k] = np.dot(RM[k], pos[k].T).T
    return pos

def vnorm_rp(vectors):
    """
    Normalize an array of vectors

    Arguments:
      -vectors : A 2D ARRAY of vectors to be normalized
    """

    return (vectors.T / np.sqrt(np.sum(vectors ** 2.0, axis=1))).T
