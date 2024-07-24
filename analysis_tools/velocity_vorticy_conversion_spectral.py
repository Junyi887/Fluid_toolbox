'''
Fast Poisson solver using FFT for periodic boundary conditions
https://math.stackexchange.com/questions/1809871/solve-poisson-equation-using-fft


@author: Junyi Guo
'''
import torch.nn as nn
import torch
import math
def periodic_bc(u:torch.Tensor):
    
    '''
    assign periodic boundary condition in physical space
    
    Inputs
    ------
    u : solution field
    
    Output
    ------
    u : solution field with periodic boundary condition applied
    '''    
    if len(u.size()) == 2:
        nx, ny = u.size()
        sol = torch.zeros((nx+1,ny+1))
        sol[:-1, :-1] = u[1:, 1: ]
        sol[-1, -1] = u[0,0]
        sol[-1, 1:] = u[0, 1:]
        sol[1:,-1]= u[1:,0]

    elif len(u.size()) == 3:
        T, nx, ny = u.size()
        sol = torch.zeros((T,nx,ny))
        sol[:,:-1, :-1] = u[:,1:, 1: ]
        sol[:,-1, -1] = u[:,0,0]
        sol[:,-1, 1:] = u[:,0, 1:]
        sol[:,1:,-1]= u[:,1:,0]


    return sol

def vorticity2uv(w0):
    """
    Applies the Fast Fourier Transform (FFT) to solve the Poisson equation in Fourier space. Solve the partial derivative of the stream function in Fourier space to obtain the velocity field in physical space. The boundary contidion has to be periodic.

    Args:
        w0 (torch.Tensor): The initial vorticity at physical space.

    Returns:
        velocities in physical space.

    """
    N = w0.size()[-1]
    k_max = math.floor(N/2.0)   #Maximum frequency
    #Initial vorticity to Fourier space
    w_h = torch.fft.rfft2(w0)

    #Wavenumbers in y-direction
    k_y = torch.cat((torch.arange(start=0, end=k_max, step=1, device=w0.device), torch.arange(start=-k_max, end=0, step=1, device=w0.device)), 0).repeat(N,1)
    #Wavenumbers in x-direction
    k_x = k_y.transpose(0,1)
    #Truncate redundant modes
    k_x = k_x[..., :k_max + 1]
    k_y = k_y[..., :k_max + 1]

    k2 = (k_x**2 + k_y**2) # based on the scheme used for discrtetizing the Poisson equation. This domain size is 2*pi

    k2[0,0] = 1 # avoid division by zero

    # solve Poisson equation in Fourier spaces
    # stream function in Fourier space
    psi_h = w_h / k2 # note the negative sign is cancelled out on the denominator

    u_h =  k_y * 1j * psi_h
    uu = torch.fft.irfft2(u_h, s=(N, N))
    v_h = -k_x * 1j * psi_h
    vv = torch.fft.irfft2(v_h, s=(N, N))
    return uu,vv 

def uv2vorticity(u,v):
    '''
    Calculate vorticity from velocity field in physical space using FFT.
    Note the domain size has to be 2*pi and 
    '''
    N = u.size()[-1]
    k_max = math.floor(N/2.0)   #Maximum frequency
    #Initial vorticity to Fourier space
    u_h = torch.fft.rfft2(u)
    v_h = torch.fft.rfft2(v)
    #Wavenumbers in y-direction
    k_y = torch.cat((torch.arange(start=0, end=k_max, step=1, device=u_h.device), torch.arange(start=-k_max, end=0, step=1, device=u_h.device)), 0).repeat(N,1)
    #Wavenumbers in x-direction
    k_x = k_y.transpose(0,1)
    #Truncate redundant modes
    k_x = k_x[..., :k_max + 1]
    k_y = k_y[..., :k_max + 1]
    w_h = k_x * 1j* v_h - k_y * 1j* u_h
    w = torch.fft.irfft2(w_h, s=(N, N))

    # periodic boundary condition
    w = periodic_bc(w)
    return w
