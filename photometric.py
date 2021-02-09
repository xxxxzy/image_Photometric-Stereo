import sys
import matplotlib.pyplot  as plt
import numpy as np
from scipy import fftpack as fft
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
import random
#from mayavi import mlab



def read_data_file(filename):

    from scipy.io import loadmat
    
    data = loadmat(filename)
    I = data['I']
    mask = data['mask']
    S = data['S']
    return I, mask, S


def cdx(f):
    """
    central differences for f in x direction
    """
    m = f.shape[0]
    west = [0] + list(range(m-1))
    east = list(range(1,m)) + [m-1]
    return 0.5*(f[east,:] - f[west,:])


def cdy(f):
    """
    central differences for f in y direction
    """
    n = f.shape[1]
    south = [0] + list(range(n-1))
    north = list(range(1,n)) + [n-1]
    return 0.5*(f[:,north] - f[:,south])
    

def tolist(A):
    """
    Linearize array to a 1D list
    """
    return list(np.reshape(A, A.size))


def unbiased_integrate(n1, n2, n3, mask, order=2):
    
    
    p = -n2/n3
    q = -n1/n3
    
    # Calculate some usefuk masks
    m,n = mask.shape
    Omega = np.zeros((m,n,4))
    Omega_padded = np.pad(mask, (1,1), mode='constant', constant_values=0)
    Omega[:, :, 0] = Omega_padded[2:, 1:-1]*mask   # value 1 iff bottom neighbor also in mask
    Omega[:, :, 1] = Omega_padded[:-2, 1:-1]*mask  # value 1 iff top neighbor also in mask
    Omega[:, :, 2] = Omega_padded[1:-1, 2:]*mask   # value 1 iff right neighbor also in mask
    Omega[:, :, 3] = Omega_padded[1:-1, :-2]*mask  # value 1 iff left neighbor also in mask
    del Omega_padded
    
    # Mapping between 2D indices and an linear indices of
    # pixels inside the mask
    indices_mask = np.where(mask > 0)
    lidx = len(indices_mask[0])
    mapping_matrix = np.zeros(p.shape, dtype=int)
    mapping_matrix[indices_mask] = list(range(lidx))
    
    if order == 1:
        pbar = p.copy()
        qbar = q.copy()
    elif order == 2:
        pbar = 0.5*(p + p[list(range(1,m)) + [m-1], :])  # p <- (p + south(p))/2
        qbar = 0.5*(q + q[:, list(range(1,n)) + [n-1]])  # q <- (q + east(q))/2
        
    # System
    I = []
    J = []
    K = []
    b = np.zeros(lidx)

    # In mask, right neighbor also in mask
    rset = Omega[:,:,2]
    X, Y = np.where(rset > 0)
    I_center = mapping_matrix[(X,Y)].astype(int)
    I_neighbors = mapping_matrix[(X,Y+1)]
    lic = len(X)
    A_center = np.ones(lic)
    A_neighbors = -A_center
    K += tolist(A_center) + tolist(A_neighbors)
    I += tolist(I_center) + tolist(I_center)
    J += tolist(I_center) + tolist(I_neighbors)
    b[I_center] -= qbar[(X,Y)]

    # In mask, left neighbor in mask
    lset = Omega[:,:,3]
    X, Y = np.where(lset > 0)
    I_center = mapping_matrix[(X,Y)].astype(int)
    I_neighbors = mapping_matrix[(X,Y-1)]
    lic = len(X)
    A_center = np.ones(lic)
    A_neighbors = -A_center
    K += tolist(A_center) + tolist(A_neighbors)
    I += tolist(I_center) + tolist(I_center)
    J += tolist(I_center) + tolist(I_neighbors)
    b[I_center] += qbar[(X,Y-1)]

    # In mask, top neighbor in mask
    tset = Omega[:,:,1]
    X, Y = np.where(tset > 0)
    I_center = mapping_matrix[(X,Y)].astype(int)
    I_neighbors = mapping_matrix[(X-1,Y)]
    lic = len(X)
    A_center = np.ones(lic)
    A_neighbors = -A_center
    K += tolist(A_center) + tolist(A_neighbors)
    I += tolist(I_center) + tolist(I_center)
    J += tolist(I_center) + tolist(I_neighbors)
    b[I_center] += pbar[(X-1,Y)]

    #    In mask, bottom neighbor in mask
    bset = Omega[:,:,0]
    X, Y = np.where(bset > 0)
    I_center = mapping_matrix[(X,Y)].astype(int)
    I_neighbors = mapping_matrix[(X+1,Y)]
    lic = len(X)
    A_center = np.ones(lic)
    A_neighbors = -A_center
    K += tolist(A_center) + tolist(A_neighbors)
    I += tolist(I_center) + tolist(I_center)
    J += tolist(I_center) + tolist(I_neighbors)
    b[I_center] -= pbar[(X,Y)]
    
    # Construction de A (compressed sparse column matrix)
    A = sp.csc_matrix((K, (I, J)))
    A = A + sp.eye(A.shape[0])*1e-9
    z = np.nan*np.ones(mask.shape)
    z[indices_mask] = spsolve(A, b)
    return z
    

def simchony_integrate(n1, n2, n3, mask):
    
    m,n = n1.shape
    p = -n2/n3
    q = -n1/n3

    outside = np.where(mask == 0)
    p[outside] = 0
    q[outside] = 0

    # divergence of (p,q)
    px = cdx(p)
    qy = cdy(q)
    
    f = px + qy

    # 4 edges
    f[0,1:-1]  = 0.5*(p[0,1:-1] + p[1,1:-1])
    f[-1,1:-1] = 0.5*(-p[-1,1:-1] - p[-2,1:-1])
    f[1:-1,0]  = 0.5*(q[1:-1,0] + q[1:-1,1])
    f[1:-1,-1] = 0.5*(-q[1:-1,-1] - q[1:-1,-2])

    # 4 corners
    f[ 0, 0] = 0.5*(p[0,0] + p[1,0] + q[0,0] + q[0,1])
    f[-1, 0] = 0.5*(-p[-1,0] - p[-2,0] + q[-1,0] + q[-1,1])
    f[ 0,-1] = 0.5*(p[0,-1] + p[1,-1] - q[0,-1] - q[1,-1])
    f[-1,-1] = 0.5*(-p[-1,-1] - p[-2,-1] -q[-1,-1] -q[-1,-2])

    # cosine transform f (reflective conditions, a la matlab,
    # might need some check)
    fs = fft.dctn(f, norm='ortho')
    
    x, y = np.mgrid[0:m,0:n]
    denum = (2*np.cos(np.pi*x/m) - 2) + (2*np.cos(np.pi*y/n) -2)
    Z = fs/denum
    Z[0,0] = 0.0
    # or what Yvain proposed, it does not really matters
    # Z[0,0] = Z[1,0] + Z[0,1]
    z = fft.idctn(Z, norm='ortho')
    # fill outside with Nan, i.e., undefined.
    z[np.where(mask == 0)] = np.nan
    return z
    
 
 
def AlbedoDepth(File,num_img):
    
    I = File[0]
    mask = File[1]
    S = File[2]

    
    '''
    output nonzero index in mask；
    type：“tuple”;its 2d array，so it will output 2 arrays. 
          the first is index of line，second is colum
          valid=(line,colum)
    '''
    a = np.where(mask)
    line = a[0]
    colum = a[1]
    
    omask = list(np.array(mask).flatten()) #transfer 1d list
    nonzero = omask.count(1) #the number of nonzero parts of mask    
    
    J=np.zeros((num_img,nonzero)) #initial J
    
    '''
    Define J with nonzero part in each image of I
    '''
    for i in range (0,num_img):
        for j in range (0,nonzero):
            J[i,j]=I[line[j],colum[j],i]
    #print(J)
    
    if num_img>3:
        
        S_pinv=np.linalg.pinv(S)
        M=np.dot(S_pinv,J) #M=S^(-1)J; M.shape=(3,28898)
        
    else:
        
        S_inv=np.linalg.inv(S)
        M=np.dot(S_inv,J)     
    #print(M)
    
    m1 = M[0]
    m2 = M[1]
    m3 = M[2]
    
    albedo = np.sqrt(m1**2+m2**2+m3**2)
    #print(albedo)
    
    albedo_msk = np.zeros(mask.shape)
    for i in range (0,nonzero):
        albedo_msk[line[i],colum[i]]=albedo[i]
        
    
    plt.axis('off')
    plt.imshow(albedo_msk,cmap='gray')
    plt.show()
    
    
    '''
    normalized M: M_norm = M/||M||
    '''
    M_nor = M/albedo
    #print(M_nor[1])
    n1 = M_nor[0]
    n2 = M_nor[1]
    n3 = M_nor[2]

    
    '''
    calculate n1,n2,n3
    '''
    mask_n1 = np.zeros(mask.shape)
    
    for i in range (0,nonzero):
        mask_n1[line[i],colum[i]]=n1[i]

    #print(mask_n1)
    
    
    mask_n2 = np.zeros(mask.shape)
    
    for i in range (0,nonzero):
        mask_n2[line[i],colum[i]]=n2[i]
    
    #print(mask_n2)
    

    mask_n3 = np.zeros(mask.shape)
    
    for i in range (0,nonzero):
        mask_n3[line[i],colum[i]]=n3[i]
    
    #print(mask_n3)
    
    
    '''
    Compute z with two functions : nbiased_integrate and simchony_integrate
    '''
    

    z_unbiase = unbiased_integrate(mask_n1, mask_n2, mask_n3, mask, order=2)
    z1=np.nan_to_num(z_unbiase) #transfer nan to 0
    
    z_simchony = simchony_integrate(mask_n1, mask_n2, mask_n3, mask)
    z2 = np.nan_to_num(z_simchony)

    
    return
    


def ransac_3dvector(data, threshold, max_data_tries=100, max_iters=1000,
                    p=0.9, det_threshold=1e-1, verbose=2):


    # minimum number of  points to select a model
    global s
    n_model_points = 3
    
    # initialisation of model to None,
    best_m = None
    
    # a count of attempts at selecting a good data set
    trial_count = 0
    
    # score of currently best selected model
    best_score = 0
    
    # best average fit for a model
    best_fit = float("Inf")
    
    # number of trials needed to pick a correct a correct dataset
    # with probability p. Will be updated during the run (same name
    # as in Fishler-Bolles.
    k = 1
    
    I, S = data
    #S = S.T
    S = S.copy().astype(float)
    I = I.copy().astype(float)
    ndata = len(I)
    print(S)
    
    while k > trial_count:
        if verbose >= 2:
            print("ransac_3dvector(): at trial ",trial_count)

        i = 0
        while i < max_data_tries:
            # select 3 pairs s, I randomly and check whether
            # they allow to compute a proper model: |det(s_i1, s_i2, s_i3| >> 0.
            idx = random.sample(range(ndata), n_model_points)
            if verbose >= 2:
                print("ransac_3dvector(): selected indices = ", idx)
            s = S[idx]
            if abs(np.linalg.det(s))>= det_threshold:
                Is = I[idx]
                break
            i += 1
        if i == max_data_tries:
            if verbose >= 1:
                print("ransac_3dvector(): no dataset found, degenerate model?")
            return None
        
        # here, we can evaluate a candidate model
        m = np.linalg.inv(s) @ Is

        if verbose >= 2:
            print("ransac_3dvector(): estimated model", m)
        # then its inliers. For that we fist compute fitting values
        fit = np.abs(I - S @ m)
        inliers = np.where(fit <= threshold)[0]
        n_inliers = len(inliers)
        if verbose >= 2:
            print("ransac_3dvector(): number of inliers for this model", n_inliers)

        
        if n_inliers > best_score:
            best_score = n_inliers
            best_inliers = inliers
            # we reevaluate m on the inliers' subset
            s = S[inliers]
            Is = I[inliers]
            best_m = np.linalg.pinv(s) @ Is
            # This should match Yvain's version?
            # best_m = m.copy()
            best_fit = np.mean(np.abs(Is - s@best_m))
            if verbose >= 2:
                print("ransac_3dvector(), updating best model to", best_m)
            
            frac_inliers = n_inliers / ndata
            # p_outliers is the 1 - b of Fishler-Bolles
            # the number of needed points to select a model is 3
            p_outliers = 1 - frac_inliers**n_model_points
            # preveny NaN/Inf  in estimation of k
            eps = np.spacing(p_outliers)
            p_outliers = min(1-eps, max(eps, p_outliers))
            k = np.log(1-p)/np.log(p_outliers)
            if verbose >= 2:
                print("ransac_3dvector(): estimate of runs to select"
                      " enough inliers with probability {0}: {1}".format(p, k))

        trial_count += 1
        if trial_count > max_iters:
            if verbose:
                print("ransac_3dvector(): reached maximum number of trials.")
            break

    if best_m is None:
        if verbose:
            print("ransac_3dvector(): unable to find a good enough solution.")
        return None
    else:
        if verbose >= 2:
            print("ransac_3dvector(): returning after {0} iterations.".format(trial_count))
        return best_m, best_inliers, best_fit





def make_bc_data(mask):

    
    m,n = mask.shape
    inside = np.where(mask)
    x, y = inside
    n_pixels = len(x)
    m2i = -np.ones(mask.shape)
    # m2i[i,j] = -1 if (i,j) not in domain, index of (i,j) else.
    m2i[(x,y)] = range(n_pixels)
    west  = np.zeros(n_pixels, dtype=int)
    north = np.zeros(n_pixels, dtype=int)
    east  = np.zeros(n_pixels, dtype=int)
    south = np.zeros(n_pixels, dtype=int)

   
    for i in range(n_pixels):
        xi = x[i]
        yi = y[i]
        wi = x[i] - 1
        ni = y[i] + 1
        ei = x[i] + 1
        si = y[i] - 1

        west[i]  = m2i[wi,yi] if (wi > 0) and (mask[wi, yi] > 0) else i
        north[i] = m2i[xi,ni] if (ni < n) and (mask[xi, ni] > 0) else i
        east[i]  = m2i[ei,yi] if (ei < m) and (mask[ei, yi] > 0) else i
        south[i] = m2i[xi,si] if (si > 0) and (mask[xi, si] > 0) else i

    return west, north, east, south, inside, n_pixels


def project_orthogonal(p, n):

    
    sshape = p.shape[:-1] + (1,)
    h = (p*n).sum(axis=-1)
    return p - n*h.reshape(sshape)

def sphere_Exp_map(v, n, eps=1e-7):

    
    sshape = v.shape[:-1] + (1,)
    nv = np.linalg.norm(v, axis=-1)
    cv = np.cos(nv).reshape(sshape)
    sv = np.sin(nv).reshape(sshape)
    
    # to avoid division by 0, when |nv| is < eps, replace by 1
    # do not change the evolution!
    np.place(nv, nv < eps, 1.0)
    vhat = v/nv.reshape(sshape)
    return n*cv + vhat*sv


def smooth_normal_field(n1, n2, n3, mask, bc_list=None, iters=100, tau=0.05, verbose=False):

    if bc_list is None:
        bc_list = make_bc_data(mask)
    west, north, east, south, inside, n_pixels = bc_list
    N = np.zeros((n_pixels, 3))
    N[:,0] = n1[inside]
    N[:,1] = n2[inside]
    N[:,2] = n3[inside]

    for i in range(iters):
        if verbose:
            sys.stdout.write(f'\rsmoothing iteration {i} out of {iters}\t')
        # Tension (a.k.a vector-valued Laplace Beltrami on proper bundle)
        v3 = N[west] + N[north] + N[east] + N[south] - 4.0*N

        grad = project_orthogonal(v3, N)
        # Riemannian Exponential map for evolution
        N = sphere_Exp_map(tau*grad, N)

    if verbose:
        print('\n')

    N = N.T
    N1 = np.zeros(mask.shape)
    N2 = np.zeros(mask.shape)
    N3 = np.ones(mask.shape)

    N1[inside] = N[0]
    N2[inside] = N[1]
    N3[inside] = N[2]
    return N1, N2, N3
    
    
    
def Ransac(File,num_img,threshold):
    
    I = File[0]
    mask = File[1]
    S = File[2]
    
    a = np.where(mask)
    line = a[0]
    colum = a[1]
    
    omask = list(np.array(mask).flatten())
    nz = omask.count(1)
        
    J=np.zeros((num_img,nz)) #initial J
    

    for i in range (0,num_img):
        for j in range (0,nz):
            J[i,j]=I[line[j],colum[j],i]
    
    '''
    estimate M by Ransac
    '''
    
    M = np.zeros((3,nz))

    for i in range (0,nz):
        
        data = (J[:,i],S)
        num = ransac_3dvector(data, threshold, max_data_tries=100, max_iters=1000, p=0.9, det_threshold=1e-1, verbose=2)
        M[:,i] = num[0]
    #print(M)
    
    '''
    albedo
    '''
    m1 = M[0]
    m2 = M[1]
    m3 = M[2]
    
    albedo = np.sqrt(m1**2+m2**2+m3**2)    
    
    albedo_msk = np.zeros(mask.shape)
    
    for i in range (0,nz):
        albedo_msk[line[i],colum[i]]=albedo[i]
        
    
    plt.axis('off')
    plt.imshow(albedo_msk,cmap='gray')
    plt.show()
    
    N = M/albedo
    #print(M_nor[1])
    n1 = N[0]
    n2 = N[1]
    n3 = N[2]

    
    '''
    normal depth
    '''
    mask_n1 = np.zeros(mask.shape)
    
    for i in range (0,nz):
        mask_n1[line[i],colum[i]]=n1[i]

    #print(mask_n1)
    
    
    mask_n2 = np.zeros(mask.shape)
    
    for i in range (0,nz):
        mask_n2[line[i],colum[i]]=n2[i]
    
    #print(mask_n2)
    

    mask_n3 = np.zeros(mask.shape)
    
    for i in range (0,nz):
        mask_n3[line[i],colum[i]]=n3[i]
    
    #print(mask_n3)
    z_unbiase = unbiased_integrate(mask_n1, mask_n2, mask_n3, mask, order=2)
    z1=np.nan_to_num(z_unbiase) #transfer nan to 0
    
    z_simchony = simchony_integrate(mask_n1, mask_n2, mask_n3, mask)
    z2 = np.nan_to_num(z_simchony)
    
    
    '''
    smooth depth
    '''
    
    field = smooth_normal_field(mask_n1, mask_n2, mask_n3, mask, bc_list=None, iters=100, tau=0.05, verbose=False)
    N1 = field[0]
    N2 = field[1]
    N3 = field[2]

    z_smoothunbiase = unbiased_integrate(N1, N2, N3, mask, order=2)
    z3=np.nan_to_num(z_smoothunbiase) #transfer nan to 0
    
    z_smoothsimchony = simchony_integrate(N1, N2, N3, mask)
    z4 = np.nan_to_num(z_smoothsimchony)
    
    return
  





