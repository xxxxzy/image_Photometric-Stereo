import ps_utils as ps
#from mayavi import mlab
import numpy as np
#from matplotlib import pyplot as plt
import matplotlib.pyplot  as plt




Beethoven = ps.read_data_file('Beethoven')
mat_vase = ps.read_data_file('mat_vase')
shiny_vase = ps.read_data_file('shiny_vase')
shiny_vase2 = ps.read_data_file('shiny_vase2')
Buddha = ps.read_data_file('Buddha')
face = ps.read_data_file('face')


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
    

    z_unbiase = ps.unbiased_integrate(mask_n1, mask_n2, mask_n3, mask, order=2)
    z1=np.nan_to_num(z_unbiase) #transfer nan to 0
    ps.display_surface(z1, albedo=None)
    
    z_simchony = ps.simchony_integrate(mask_n1, mask_n2, mask_n3, mask)
    z2 = np.nan_to_num(z_simchony)
    ps.display_surface(z2, albedo=None)

    
    return
    
    
AlbedoDepth(Beethoven,3)   
AlbedoDepth(mat_vase,3)
AlbedoDepth(shiny_vase,3)
AlbedoDepth(shiny_vase2,22)
AlbedoDepth(Buddha,10)
AlbedoDepth(face,27)


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
        num = ps.ransac_3dvector(data, threshold, max_data_tries=100, max_iters=1000, p=0.9, det_threshold=1e-1, verbose=2)
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
    z_unbiase = ps.unbiased_integrate(mask_n1, mask_n2, mask_n3, mask, order=2)
    z1=np.nan_to_num(z_unbiase) #transfer nan to 0
    ps.display_surface(z1, albedo=None)
    
    z_simchony = ps.simchony_integrate(mask_n1, mask_n2, mask_n3, mask)
    z2 = np.nan_to_num(z_simchony)
    ps.display_surface(z2, albedo=None)
    
    
    '''
    smooth depth
    '''
    
    field = ps.smooth_normal_field(mask_n1, mask_n2, mask_n3, mask, bc_list=None, iters=100, tau=0.05, verbose=False)
    N1 = field[0]
    N2 = field[1]
    N3 = field[2]

    z_smoothunbiase = ps.unbiased_integrate(N1, N2, N3, mask, order=2)
    z3=np.nan_to_num(z_smoothunbiase) #transfer nan to 0
    ps.display_surface(z3, albedo=None)
    
    z_smoothsimchony = ps.simchony_integrate(N1, N2, N3, mask)
    z4 = np.nan_to_num(z_smoothsimchony)
    ps.display_surface(z4, albedo=None)
    
    return

Ransac(shiny_vase,3,1)
Ransac(shiny_vase2,22,1)
Ransac(Buddha,10,25) #125跑的要快一点
Ransac(face,27,10)     





