import torch

def energy_spectrum(nx,ny,w):
    '''
    Note the domain size has to be 2*pi, otherwise the energy spectrum will be wrong.
    
    '''
    epsilon = 1.0e-6

    kx = torch.empty(nx)
    ky = torch.empty(ny)
    dx = 2.0*torch.pi/nx
    dy = 2.0*torch.pi/ny
    kx[0:int(nx/2)] = 2*torch.pi/(nx*dx)*torch.arange(0,int(nx/2))
    kx[int(nx/2):nx] = 2*torch.pi/(nx*dx)*torch.arange(-int(nx/2),0)

    ky[0:ny] = kx[0:ny]
    
    kx[0] = epsilon
    ky[0] = epsilon

    kx, ky = torch.meshgrid(kx, ky, indexing='ij')
    
    w = torch.tensor(w)
    print(w.shape)
    wf = torch.fft.fft2(w)
    print(wf.shape)
    es =  torch.empty((nx,ny))
    print(es.shape)

    kk = torch.sqrt(kx[:,:]**2 + ky[:,:]**2)

    print(kk.shape)
    es[:,:] = torch.pi*((torch.abs(wf[:,:])/(nx*ny))**2)/kk
    
    n = int(np.sqrt(nx*nx + ny*ny)/2.0)-1
    
    en = torch.zeros(n+1)
    
    for k in range(1,n+1):
        en[k] = 0.0
        ic = 0
        ii,jj = torch.where((kk[1:,1:]>(k-0.5)) & (kk[1:,1:]<(k+0.5)))
        ic = ii.numel()
        ii = ii+1
        jj = jj+1
        en[k] = torch.sum(es[ii,jj])
                    
        en[k] = en[k]/ic
        
    return en, n