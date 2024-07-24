"""
Ref: https://doi.org/10.48550/arXiv.2201.02928
Two-dimensional Navier-Stokes solver  
Vorticity-stream function formulation
Arakawa scheme (or compact scheme or explicit upwind) for nonlinear term
3rd order Runge-Kutta for temporal discritization
Periodic boundary conditions only

"""
import numpy as np
from numpy.random import seed
import pyfftw
from scipy import integrate
from scipy import linalg
import matplotlib.pyplot as plt 
import time as tm
import matplotlib.ticker as ticker
import os
# from numba import jit

# from tensorflow.keras.models import Sequential, Model, load_model
# from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
# from tensorflow.keras import backend as K

print('tf package removed...')

from scipy import ndimage
from scipy.ndimage import gaussian_filter
import yaml
import argparse
import os
import sys

from utils import *

# font = {'size'   : 14}    
# plt.rc('font', **font)
# import matplotlib as mpl
# mpl.rcParams['text.usetex'] = True
# mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

#%%
# def coeff_determination(y_true, y_pred):
#     SS_res =  K.sum(K.square( y_true-y_pred ))
#     SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
#     return ( 1 - SS_res/(SS_tot + K.epsilon()) )

#%%
f_use_argparse = True
if f_use_argparse:
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default="config/input.yaml",
                        required=True, help="Input yaml file")
    args = parser.parse_args()
    config_file = args.config
else:
    config_file = "config/input.yaml"

with open(config_file) as file:
    input_data = yaml.load(file, Loader=yaml.FullLoader)
file.close()

nd = input_data['nd']
nt = input_data['nt']
re = np.float32(input_data['re'])
dt = input_data['dt']
ns = input_data['ns']
isolver = input_data['isolver']
ifm = input_data['ifm']
ipr = input_data['ipr']
istart = input_data['istart']
kappa = input_data['kappa']
pCU3 = input_data['pCU3']
n_dns = input_data['n_dns']
seed_number = input_data['seed_number']
ic_folder = input_data['ic_folder']
ic_num_snapshot = input_data['ic_num_snapshot']
iprec = input_data['iprec']
ifeat = 2

print('Simulate NS KT equation with Re = ', re)
print('The # snapshots: ', ns)

seed(seed_number)

#%%
freq = int(nt/ns)

if ifm == 0 or ifm == 1:
    model = []
    max_min = []
    
# elif ifm == 2:
#     model = load_model('CNN.hd5',
#                        custom_objects={'coeff_determination': coeff_determination})
    
#     data = np.load('CNN_DATA.npz')
#     max_min = data['max_min']
    
#     del data
    
# elif ifm == 3:
#     if ifeat == 2:
#         model = load_model('./nn_history/DNN_model_2_'+str(ifeat)+'_G.hd5',
#                            custom_objects={'coeff_determination': coeff_determination})
#     elif ifeat == 3:
#         model = load_model('./nn_history/DNN_model_1_'+str(ifeat)+'_G.hd5',custom_objects={'coeff_determination': coeff_determination})
    
#     max_min = np.load('./nn_history/scaling_dnn_G.npy')
    
#%%
# fast poisson solver using second-order central difference scheme
def fps(nx, ny, dx, dy, f):
    epsilon = 1.0e-6
    aa = -2.0/(dx*dx) - 2.0/(dy*dy)
    bb = 2.0/(dx*dx)
    cc = 2.0/(dy*dy)
    hx = 2.0*np.pi/np.float64(nx)
    hy = 2.0*np.pi/np.float64(ny)
    
    kx = np.empty(nx)
    ky = np.empty(ny)
    
    kx[:] = hx*np.float64(np.arange(0, nx))

    ky[:] = hy*np.float64(np.arange(0, ny))
    
    kx[0] = epsilon
    ky[0] = epsilon

    kx, ky = np.meshgrid(np.cos(kx), np.cos(ky), indexing='ij')
    
    data = np.empty((nx,ny), dtype='complex128')
    data1 = np.empty((nx,ny), dtype='complex128')
    
    data[:,:] = np.vectorize(complex)(f[2:nx+2,2:ny+2],0.0)

    a = pyfftw.empty_aligned((nx,ny),dtype= 'complex128')
    b = pyfftw.empty_aligned((nx,ny),dtype= 'complex128')
    
    fft_object = pyfftw.FFTW(a, b, axes = (0,1), direction = 'FFTW_FORWARD')
    fft_object_inv = pyfftw.FFTW(a, b,axes = (0,1), direction = 'FFTW_BACKWARD')
    
    e = fft_object(data)
    #e = pyfftw.interfaces.scipy_fftpack.fft2(data)
    
    e[0,0] = 0.0
    
    data1[:,:] = e[:,:]/(aa + bb*kx[:,:] + cc*ky[:,:])

    ut = np.real(fft_object_inv(data1))
    
    #periodicity
    u = np.empty((nx+5,ny+5)) 
    u[2:nx+2,2:ny+2] = ut
    u[:,ny+2] = u[:,2]
    u[nx+2,:] = u[2,:]
    u[nx+2,ny+2] = u[2,2]
    
    return u

#%%
#-----------------------------------------------------------------------------#
# Solution to tridigonal system using Thomas algorithm
#-----------------------------------------------------------------------------#
def tdmsv(a,b,c,r,s,e,n):
    gam = np.zeros((e+1,n+1))
    u = np.zeros((e+1,n+1))
    bet = np.zeros((1,n+1))
    
    bet[0,:] = b[s,:]
    u[s,:] = r[s,:]/bet[0,:]
    
    for i in range(s+1,e+1):
        gam[i,:] = c[i-1,:]/bet[0,:]
        bet[0,:] = b[i,:] - a[i,:]*gam[i,:]
        u[i,:] = (r[i,:] - a[i,:]*u[i-1,:])/bet[0,:]
    
    for i in range(e-1,s-1,-1):
        u[i,:] = u[i,:] - gam[i+1,:]*u[i+1,:]
    
    return u
        
#-----------------------------------------------------------------------------#
# Solution to tridigonal system using cyclic Thomas algorithm
#-----------------------------------------------------------------------------#
def ctdmsv(a,b,c,alpha,beta,r,s,e,n):
    bb = np.zeros((e+1,n+1))
    u = np.zeros((e+1,n+1))
    gamma = np.zeros((1,n+1))
    
    gamma[0,:] = -b[s,:]
    bb[s,:] = b[s,:] - gamma[0,:]
    bb[e,:] = b[e,:] - alpha*beta/gamma[0,:]
    
#    for i in range(s+1,e):
#        bb[i] = b[i]
    
    bb[s+1:e,:] = b[s+1:e,:]
    
    x = tdmsv(a,bb,c,r,s,e,n)
    
    u[s,:] = gamma[0,:]
    u[e,:] = alpha[0,:]
    
    z = tdmsv(a,bb,c,u,s,e,n)
    
    fact = (x[s,:] + beta[0,:]*x[e,:]/gamma[0,:])/(1.0 + z[s,:] + beta[0,:]*z[e,:]/gamma[0,:])
    
#    for i in range(s,e+1):
#        x[i] = x[i] - fact*z[i]
    
    x[s:e+1,:] = x[s:e+1,:] - fact*z[s:e+1,:]
        
    return x

#-----------------------------------------------------------------------------#
#cu3dp: 3rd-order compact upwind scheme for the first derivative(up)
#       periodic boundary conditions (0=n), h=grid spacing
#       p: free upwind paramater suggested (p>0 for upwind)
#                                           p=0.25 in Zhong (JCP 1998)
#		
#-----------------------------------------------------------------------------#
def cu3dpv(u,p,h,nx,ny):
    a = np.zeros((nx,ny+1))
    b = np.zeros((nx,ny+1))        
    c = np.zeros((nx,ny+1))    
    x = np.zeros((nx,ny+1))
    r = np.zeros((nx,ny+1))  
    up = np.zeros((nx+1,ny+1))
    
    a[:,:] = 1.0 + p
    b[:,:] = 4.0
    c[:,:] = 1.0 - p

#    for i in range(1,n):
#        r[i] = ((-3.0-2.0*p)*u[i-1] + 4.0*p*u[i] + (3.0-2.0*p)*u[i+1])/h
    r[1:nx,:] = ((-3.0-2.0*p)*u[0:nx-1,:] + 4.0*p*u[1:nx,:] + (3.0-2.0*p)*u[2:nx+1,:])/h
    r[0,:] = ((-3.0-2.0*p)*u[nx-1,:] + 4.0*p*u[0,:] + (3.0-2.0*p)*u[1,:])/h  
    
    alpha = np.zeros((1,nx+1))
    beta = np.zeros((1,nx+1))
    
    alpha[0,:] = 1.0 - p
    beta[0,:] = 1.0 + p
    
    x = ctdmsv(a,b,c,alpha,beta,r,0,nx-1,ny)
    
    up[0:nx,:] = x[0:nx,:]
    
    up[nx,:] = up[0,:]
    
    return up

#-----------------------------------------------------------------------------#
# c4dp:  4th-order compact scheme for first-degree derivative(up)
#        periodic boundary conditions (0=n), h=grid spacing
#        tested
#		
#-----------------------------------------------------------------------------#
def c4dpv(u,h,nx,ny):
    a = np.zeros((nx,ny+1))
    b = np.zeros((nx,ny+1))        
    c = np.zeros((nx,ny+1))    
    x = np.zeros((nx,ny+1))
    r = np.zeros((nx,ny+1))  
    up = np.zeros((nx+1,ny+1))
    
    a[:,:] = 1.0/4.0
    b[:,:] = 1.0
    c[:,:] = 1.0/4.0

#    for i in range(1,n):
#        r[i] = (3.0/2.0)*(u[i+1] - u[i-1])/(2.0*h)
    r[1:nx,:] = (3.0/2.0)*(u[2:nx+1,:] - u[0:nx-1,:])/(2.0*h)
    r[0,:] = (3.0/2.0)*(u[1,:] - u[nx-1,:])/(2.0*h)
    
    alpha = np.zeros((1,ny+1))
    beta = np.zeros((1,ny+1))
    
    alpha[0,:] = 1.0/4.0
    beta[0,:] = 1.0/4.0
    
    x = ctdmsv(a,b,c,alpha,beta,r,0,nx-1,ny)
    
    up[0:nx,:] = x[0:nx,:]
    
    up[nx,:] = up[0,:]
    
    return up

#-----------------------------------------------------------------------------#
# c4ddp:  4th-order compact scheme for first-degree derivative(up)
#        periodic boundary conditions (0=n), h=grid spacing
#        tested
#		
#-----------------------------------------------------------------------------#
def c4ddpv(u,h,nx,ny):
    a = np.zeros((nx,ny+1))
    b = np.zeros((nx,ny+1))        
    c = np.zeros((nx,ny+1))    
    x = np.zeros((nx,ny+1))
    r = np.zeros((nx,ny+1))  
    upp = np.zeros((nx+1,ny+1))
    
    a[:,:] = 1.0/10.0
    b[:,:] = 1.0
    c[:,:] = 1.0/10.0

#    for i in range(1,n):
#        r[i] = (6.0/5.0)*(u[i-1] - 2.0*u[i] + u[i+1])/(h*h)
    
    r[1:nx,:] = (6.0/5.0)*(u[0:nx-1,:] - 2.0*u[1:nx,:] + u[2:nx+1,:])/(h*h)
    r[0,:] = (6.0/5.0)*(u[nx-1,:] - 2.0*u[0,:] + u[1,:])/(h*h)
    
    alpha = np.zeros((1,ny+1))
    beta = np.zeros((1,ny+1))
    
    alpha[0,:] = 1.0/10.0
    beta[0,:] = 1.0/10.0
    
    x = ctdmsv(a,b,c,alpha,beta,r,0,nx-1,ny)
    
    upp[0:nx,:] = x[0:nx,:]
    
    upp[nx,:] = upp[0,:]
    
    return upp     
        
        
#%%
# set periodic boundary condition for ghost nodes. Index 0 and (n+2) are the ghost boundary locations
def bc(nx,ny,u):
    u[:,0] = u[:,ny]
    u[:,1] = u[:,ny+1]
    u[:,ny+3] = u[:,3]
    u[:,ny+4] = u[:,4]
    
    u[0,:] = u[nx,:]
    u[1,:] = u[nx+1,:]
    u[nx+3,:] = u[3,:]
    u[nx+4,:] = u[4,:]
    
    return u

#%%
def dnn_closure(nx,ny,w,s,max_min,model,ifeat):
    wx,wy = grad_spectral(nx,ny,w[2:nx+3,2:ny+3])
    wxx,wxy = grad_spectral(nx,ny,wx)
    wyx,wyy = grad_spectral(nx,ny,wy)
    
    sx,sy = grad_spectral(nx,ny,s[2:nx+3,2:ny+3])
    sxx,sxy = grad_spectral(nx,ny,sx)
    syx,syy = grad_spectral(nx,ny,sy)
    
    kernel_w = np.sqrt(wx**2 + wy**2)
    kernel_s = np.sqrt(4.0*sxy**2 + (sxx - syy)**2)
    
    wc = np.zeros((nx+5,ny+5))
    sc = np.zeros((nx+5,ny+5))
    kwc = np.zeros((nx+5,ny+5))
    ksc = np.zeros((nx+5,ny+5))
    
    wc[2:nx+3,2:ny+3] = (2.0*w[2:nx+3,2:ny+3] - (max_min[0,0] + max_min[0,1]))/(max_min[0,0] - max_min[0,1])
    sc[2:nx+3,2:ny+3] = (2.0*s[2:nx+3,2:ny+3] - (max_min[1,0] + max_min[1,1]))/(max_min[1,0] - max_min[1,1])
    kwc[2:nx+3,2:ny+3] = (2.0*kernel_w - (max_min[2,0] + max_min[2,1]))/(max_min[2,0] - max_min[2,1])
    ksc[2:nx+3,2:ny+3] = (2.0*kernel_s - (max_min[3,0] + max_min[3,1]))/(max_min[3,0] - max_min[3,1])
    
    wc = bc(nx,ny,wc)
    sc = bc(nx,ny,sc)
    kwc = bc(nx,ny,kwc)
    ksc = bc(nx,ny,ksc)
    
    if ifeat == 3:
        wcx = (2.0*wx - (max_min[4,0] + max_min[4,1]))/(max_min[4,0] - max_min[4,1])
        wcy = (2.0*wy - (max_min[5,0] + max_min[5,1]))/(max_min[5,0] - max_min[5,1])
        wcxx = (2.0*wxx - (max_min[6,0] + max_min[6,1]))/(max_min[6,0] - max_min[6,1])
        wcyy = (2.0*wyy - (max_min[7,0] + max_min[7,1]))/(max_min[7,0] - max_min[7,1])
        wcxy = (2.0*wxy - (max_min[8,0] + max_min[8,1]))/(max_min[8,0] - max_min[8,1])
        
        scx = (2.0*sx - (max_min[9,0] + max_min[9,1]))/(max_min[9,0] - max_min[9,1])
        scy = (2.0*sy - (max_min[10,0] + max_min[10,1]))/(max_min[10,0] - max_min[10,1])
        scxx = (2.0*sxx - (max_min[11,0] + max_min[11,1]))/(max_min[11,0] - max_min[11,1])
        scyy = (2.0*syy - (max_min[12,0] + max_min[12,1]))/(max_min[12,0] - max_min[12,1])
        scxy = (2.0*sxy - (max_min[13,0] + max_min[13,1]))/(max_min[13,0] - max_min[13,1])

    if ifeat == 1:
        nt = int((nx+1)*(ny+1))
        x_test = np.zeros((nt,18))
        n = 0
        for i in range(2,nx+3):
            for j in range(2,ny+3):
                x_test[n,0:9] = wc[i-1:i+2,j-1:j+2].flatten()
                x_test[n,9:18] = sc[i-1:i+2,j-1:j+2].flatten()
                n = n+1         
    if ifeat == 2:
        nt = int((nx+1)*(ny+1))
        x_test = np.zeros((nt,20))
        n = 0
        for i in range(2,nx+3):
            for j in range(2,ny+3):
                x_test[n,0:9] = wc[i-1:i+2,j-1:j+2].flatten()
                x_test[n,9:18] = sc[i-1:i+2,j-1:j+2].flatten()
                x_test[n,18] = kwc[i,j]
                x_test[n,19] = ksc[i,j]
                n = n+1 
                
    if ifeat == 3:
        nt = int((nx+1)*(ny+1))
        x_test = np.zeros((nt,12))
        
        x_test[:,0] = wc[2:nx+3,2:ny+3].flatten()
        x_test[:,1] = sc[2:nx+3,2:ny+3].flatten()
        x_test[:,2] = wcx.flatten()
        x_test[:,3] = wcy.flatten()
        x_test[:,4] = wcxx.flatten()
        x_test[:,5] = wcyy.flatten()
        x_test[:,6] = wcxy.flatten()
        x_test[:,7] = scx.flatten()
        x_test[:,8] = scy.flatten()
        x_test[:,9] = scxx.flatten()
        x_test[:,10] = scyy.flatten()
        x_test[:,11] = scxy.flatten()

    y_pred_sc = model.predict(x_test)
    y_pred = 0.5*(y_pred_sc*(max_min[14,0] - max_min[14,1]) + (max_min[14,0] + max_min[14,1]))
    
    y_pred = np.reshape(y_pred,[nx+1,ny+1])
    
    return y_pred    

#%%
def cnn_closure(nx,ny,w,s,max_min,model,ifeat):
    wx,wy = grad_spectral(nx,ny,w[2:nx+3,2:ny+3])
    wxx,wxy = grad_spectral(nx,ny,wx)
    wyx,wyy = grad_spectral(nx,ny,wy)
    
    sx,sy = grad_spectral(nx,ny,s[2:nx+3,2:ny+3])
    sxx,sxy = grad_spectral(nx,ny,sx)
    syx,syy = grad_spectral(nx,ny,sy)
    
    kernel_w = np.sqrt(wx**2 + wy**2)
    kernel_s = np.sqrt(4.0*sxy**2 + (sxx - syy)**2)
    
    wc = (2.0*w[2:nx+3,2:ny+3] - (max_min[0,0] + max_min[0,1]))/(max_min[0,0] - max_min[0,1])
    sc = (2.0*s[2:nx+3,2:ny+3] - (max_min[1,0] + max_min[1,1]))/(max_min[1,0] - max_min[1,1])
    kwc = (2.0*kernel_w - (max_min[2,0] + max_min[2,1]))/(max_min[2,0] - max_min[2,1])
    ksc = (2.0*kernel_s - (max_min[3,0] + max_min[3,1]))/(max_min[3,0] - max_min[3,1])
    
    wcx = (2.0*wx - (max_min[4,0] + max_min[4,1]))/(max_min[4,0] - max_min[4,1])
    wcy = (2.0*wy - (max_min[5,0] + max_min[5,1]))/(max_min[5,0] - max_min[5,1])
    wcxx = (2.0*wxx - (max_min[6,0] + max_min[6,1]))/(max_min[6,0] - max_min[6,1])
    wcyy = (2.0*wyy - (max_min[7,0] + max_min[7,1]))/(max_min[7,0] - max_min[7,1])
    wcxy = (2.0*wxy - (max_min[8,0] + max_min[8,1]))/(max_min[8,0] - max_min[8,1])
    
    scx = (2.0*sx - (max_min[9,0] + max_min[9,1]))/(max_min[9,0] - max_min[9,1])
    scy = (2.0*sy - (max_min[10,0] + max_min[10,1]))/(max_min[10,0] - max_min[10,1])
    scxx = (2.0*sxx - (max_min[11,0] + max_min[11,1]))/(max_min[11,0] - max_min[11,1])
    scyy = (2.0*syy - (max_min[12,0] + max_min[12,1]))/(max_min[12,0] - max_min[12,1])
    scxy = (2.0*sxy - (max_min[13,0] + max_min[13,1]))/(max_min[13,0] - max_min[13,1])

    if ifeat == 1:
        x_test = np.zeros((1,nx+1,ny+1,2))
        x_test[0,:,:,0] = wc
        x_test[0,:,:,1] = sc
    if ifeat == 2:
        x_test = np.zeros((1,nx+1,ny+1,4))
        x_test[0,:,:,0] = wc
        x_test[0,:,:,1] = sc
        x_test[0,:,:,2] = kwc
        x_test[0,:,:,3] = ksc
    if ifeat == 3:
        x_test = np.zeros((1,nx+1,ny+1,12))
        x_test[0,:,:,0] = wc
        x_test[0,:,:,1] = sc
        x_test[0,:,:,2] = wcx
        x_test[0,:,:,3] = wcy
        x_test[0,:,:,4] = wcxx
        x_test[0,:,:,5] = wcyy
        x_test[0,:,:,6] = wcxy
        x_test[0,:,:,7] = scx
        x_test[0,:,:,8] = scy
        x_test[0,:,:,9] = scxx
        x_test[0,:,:,10] = scyy
        x_test[0,:,:,11] = scxy
    
    y_pred_sc = model.predict(x_test[:,:,:,:])
    y_pred = 0.5*(y_pred_sc[0,:,:,0]*(max_min[14,0] - max_min[14,1]) + (max_min[14,0] + max_min[14,1]))
    
    return y_pred
    
    
    
#%% 
# compute rhs using arakawa scheme
# computed at all physical domain points (1:nx+1,1:ny+1; all boundary points included)
# no ghost points
def rhs_arakawa(nx,ny,dx,dy,re,w,s,ifm,kappa,max_min,model,ifeat):
    aa = 1.0/(dx*dx)
    bb = 1.0/(dy*dy)
    gg = 1.0/(4.0*dx*dy)
    hh = 1.0/3.0
    
    f = np.zeros((nx+5,ny+5))
    
    #Arakawa    
    j1 = gg*( (w[3:nx+4,2:ny+3]-w[1:nx+2,2:ny+3])*(s[2:nx+3,3:ny+4]-s[2:nx+3,1:ny+2]) \
             -(w[2:nx+3,3:ny+4]-w[2:nx+3,1:ny+2])*(s[3:nx+4,2:ny+3]-s[1:nx+2,2:ny+3]))

    j2 = gg*( w[3:nx+4,2:ny+3]*(s[3:nx+4,3:ny+4]-s[3:nx+4,1:ny+2]) \
            - w[1:nx+2,2:ny+3]*(s[1:nx+2,3:ny+4]-s[1:nx+2,1:ny+2]) \
            - w[2:nx+3,3:ny+4]*(s[3:nx+4,3:ny+4]-s[1:nx+2,3:ny+4]) \
            + w[2:nx+3,1:ny+2]*(s[3:nx+4,1:ny+2]-s[1:nx+2,1:ny+2]))
    
    j3 = gg*( w[3:nx+4,3:ny+4]*(s[2:nx+3,3:ny+4]-s[3:nx+4,2:ny+3]) \
            - w[1:nx+2,1:ny+2]*(s[1:nx+2,2:ny+3]-s[2:nx+3,1:ny+2]) \
            - w[1:nx+2,3:ny+4]*(s[2:nx+3,3:ny+4]-s[1:nx+2,2:ny+3]) \
            + w[3:nx+4,1:ny+2]*(s[3:nx+4,2:ny+3]-s[2:nx+3,1:ny+2]) )

    jac = (j1+j2+j3)*hh
    
    lap = aa*(w[3:nx+4,2:ny+3]-2.0*w[2:nx+3,2:ny+3]+w[1:nx+2,2:ny+3]) \
        + bb*(w[2:nx+3,3:ny+4]-2.0*w[2:nx+3,2:ny+3]+w[2:nx+3,1:ny+2])
    
    if ifm == 0:
        f[2:nx+3,2:ny+3] = -jac + lap/re 
        
    elif ifm == 1:
        ev = dyn_smag(nx,ny,kappa,s,w)
        f[2:nx+3,2:ny+3] = -jac + lap/re + ev*lap
    
    elif ifm == 2:
        kconvolve = np.array([[1,1,1],[1,1,1],[1,1,1]])
        
        pi_source = cnn_closure(nx,ny,w,s,max_min,model,ifeat)
        nue = pi_source/lap
        
        nue_p = np.where(nue > 0, nue, 0.0)
        
#        nue_loc_avg = ndimage.generic_filter(nue_p, np.nanmean, size=3, mode='constant', cval=np.NaN)
#        nue_loc_avg = ndimage.generic_filter(nue_p, np.mean, size=3, mode='constant', cval=0.0)
        nue_loc_avg = ndimage.convolve(nue_p, kconvolve, mode='mirror')#, cval=0.0)
        nue_loc_avg = nue_loc_avg/9.0
        
#        mask1 = nue_p < nue_loc_avg
#        nue_loc_avg_use = np.where(mask1[:,:] == True, nue_p, 0.0)
        nue_loc_avg_use = np.where(nue_p < nue_loc_avg, nue_p, 0.0)
        
#        mask2 = nue_loc_avg_use > 0.0
#        pi_source = np.where(mask2[:,:] == True, pi_source[:,:], 0.0)
        pi_source = np.where(nue_loc_avg_use > 0.0, pi_source[:,:],0.0)
        
#        pi_source = cnn_closure(nx,ny,w,s,max_min,model,ifeat)
#        nue = pi_source/lap
#        #pi_source = gaussian_filter(pi_source, sigma=2)
#        pi_source = np.where(nue>=0.0*np.min(nue), pi_source[:,:],0.0)
        
        f[2:nx+3,2:ny+3] = -jac + lap/re + pi_source
    
    elif ifm == 3:
        kconvolve = np.array([[1,1,1],[1,1,1],[1,1,1]])
        
        pi_source = dnn_closure(nx,ny,w,s,max_min,model,ifeat)
        nue = pi_source/lap
        
        nue_p = np.where(nue > 0, nue, 0.0)
        
        nue_loc_avg = ndimage.convolve(nue_p, kconvolve, mode='mirror')#, cval=0.0)
        nue_loc_avg = nue_loc_avg/9.0
        
        nue_loc_avg_use = np.where(nue_p < nue_loc_avg, nue_p, 0.0)
        
        pi_source = np.where(nue_loc_avg_use > 0.0, pi_source[:,:],0.0)
        
        f[2:nx+3,2:ny+3] = -jac + lap/re + pi_source
                        
    return f

#%%
def rhs_cu3v(nx,ny,dx,dy,re,pCU3,w,s):
    lap = np.zeros((nx+5,ny+5))
    jac = np.zeros((nx+5,ny+5))
    f = np.zeros((nx+5,ny+5))

# ------------------------Laplacian-------------------------------------------#    
    # compute wxx
    a = w[2:nx+3,2:ny+3]
    wxx = c4ddpv(a,dx,nx,ny)
        
    # compute wyy
    a = w[2:nx+3,2:ny+3]        
    wyy = c4ddpv(a.T,dy,ny,nx).T 
    
    lap[2:nx+3,2:ny+3] = wxx[:,:] + wyy[:,:]

# ------------------------ Jacobian (convective term): upwind ----------------#    
    
    # sy: u
    sy = np.zeros((nx+1,ny+1))
    a = s[2:nx+3,2:ny+3]        
    sy[:,:] = c4dpv(a.T,dy,ny,nx).T
    
    # computation of wx
    wxp = np.zeros((nx+1,ny+1))
    wxn = np.zeros((nx+1,ny+1))

    a = w[2:nx+3,2:ny+3]    
    wxp[:,:] = cu3dpv(a, pCU3, dx, nx, ny) # upwind for wx
    wxn[:,:] = cu3dpv(a, -pCU3, dx, nx, ny) # downwind for wx
    
    # upwinding
    syp = np.where(sy>0,sy,0) # max(sy[i,j],0)
    syn = np.where(sy<0,sy,0) # min(sy[i,j],0)
    
    # sx: -v
    sx = np.zeros((nx+1,ny+1))
    a = s[2:nx+3,2:ny+3]
    sx[:,:] = -c4dpv(a, dx, nx, ny)
    
    # computation of wy
    wyp = np.zeros((nx+1,ny+1))
    wyn = np.zeros((nx+1,ny+1))

    a = w[2:nx+3,2:ny+3]    
    wyp[:,:] = cu3dpv(a.T, pCU3, dy, ny, nx).T # upwind for wy
    wyn[:,:] = cu3dpv(a.T, -pCU3, dy, ny, nx).T # downwind for wy  
    
    # upwinding
    sxp = np.where(sx>0,sx,0) # max(sx[i,j],0)
    sxn = np.where(sx<0,sx,0) # min(sx[i,j],0)
    
    jac[2:nx+3,2:ny+3] = (syp*wxp + syn*wxn) + (sxp*wyp + sxn*wyn)

# ------------------------ RHS -----------------------------------------------#   
    
    f[2:nx+3,2:ny+3] = -jac[2:nx+3,2:ny+3] + lap[2:nx+3,2:ny+3]/re 
    
    del sy, sx, syp, syn, sxp, sxn, wxp, wxn, wyp, wyn
    
    return f


#%%
def rhs_compactv(nx,ny,dx,dy,re,w,s):
    lap = np.zeros((nx+5,ny+5))
    jac = np.zeros((nx+5,ny+5))
    f = np.zeros((nx+5,ny+5))

# ------------------------Laplacian-------------------------------------------# 
    
    # compute wxx
    a = w[2:nx+3,2:ny+3]
    wxx = c4ddpv(a,dx,nx,ny)
    
    # compute wyy
    a = w[2:nx+3,2:ny+3]        
    wyy = c4ddpv(a.T,dy,ny,nx).T
    
    lap[2:nx+3,2:ny+3] = wxx[:,:] + wyy[:]
    
# ------------------------ Jacobian (convective term) ------------------------#    
    
    # sy
    sy = np.zeros((nx+1,ny+1))
    a = s[2:nx+3,2:ny+3]        
    sy[:,:] = c4dpv(a.T,dy,ny,nx).T
    
    # computation of wx
    wx = np.zeros((nx+1,ny+1))
    a = w[2:nx+3,2:ny+3]
    wx[:,:] = c4dpv(a,dx,nx,ny)

    # sx
    sx = np.zeros((nx+1,ny+1))
    a = s[2:nx+3,2:ny+3]
    sx[:,:] = c4dpv(a, dx, nx, ny)
    
    # computation of wy
    wy = np.zeros((nx+1,ny+1))
    a = w[2:nx+3,2:ny+3]
    wy[:,:] = c4dpv(a.T, dy, ny, nx).T
    
    jac[2:nx+3,2:ny+3] = (sy*wx - sx*wy)

# ------------------------ RHS -----------------------------------------------#   
    
    f[2:nx+3,2:ny+3] = -jac[2:nx+3,2:ny+3] + lap[2:nx+3,2:ny+3]/re
    
    del sy, wx, sx, wy
    
    return f

#%%
# compute exact solution for TGV problem
def exact_tgv(nx,ny,x,y,time,re):
    ue = np.zeros((nx+5,ny+5))
    
    nq = 4.0
    ue[2:nx+3, 2:ny+3] = 2.0*nq*np.cos(nq*x[0:nx+1, 0:ny+1])*np.cos(nq*y[0:nx+1, 0:ny+1])*np.exp(-2.0*nq*nq*time/re)
    
    ue = bc(nx,ny,ue)
    return ue

#%%
# set initial condition for TGV problem
def ic_tgv(nx,ny,x,y):
    w = np.zeros((nx+5,ny+5))
    nq = 4.0
    w[2:nx+3, 2:ny+3] = 2.0*nq*np.cos(nq*x[0:nx+1, 0:ny+1])*np.cos(nq*y[0:nx+1, 0:ny+1])
    
    w = bc(nx,ny,w)

    return w

#%%
# set initial condition for vortex merger problem
def ic_vm(nx,ny,x,y):
    w = np.zeros((nx+5,ny+5))
    sigma = np.pi
    xc1 = np.pi-np.pi/4.0
    yc1 = np.pi
    xc2 = np.pi+np.pi/4.0
    yc2 = np.pi
    
    w[2:nx+3, 2:ny+3] = np.exp(-sigma*((x[0:nx+1, 0:ny+1]-xc1)**2 + (y[0:nx+1, 0:ny+1]-yc1)**2)) \
                        + np.exp(-sigma*((x[0:nx+1, 0:ny+1]-xc2)**2 + (y[0:nx+1, 0:ny+1]-yc2)**2))
    
    w = bc(nx,ny,w)

    return w

#%%
def ic_shear(nx,ny,x,y):
    w = np.zeros((nx+5,ny+5))
    delta = 0.05
    sigma = 15/np.pi
    
#    for j in range(2,ny+3):
#        for i in range(2,nx+3):
#            if y[i-2,j-2] <= np.pi:
#                w[i,j] = delta*np.cos(x[i-2,j-2]) - sigma/  \
#                        (np.cosh(sigma*(y[i-2,j-2] - np.pi/2)))**2
#            else:
#                w[i,j] = delta*np.cos(x[i-2,j-2]) + sigma/  \
#                        (np.cosh(sigma*(3*np.pi/2 - y[i-2,j-2])))**2
    
    indy = np.array(np.where(y[0,:] <= np.pi))
    indy = indy.flatten()
    
    w[2:nx+3, indy+2] = delta*np.cos(x[0:nx+1, indy]) - sigma/  \
                        (np.cosh(sigma*(y[0:nx+1, indy] - np.pi/2)))**2
        
    indy = np.array(np.where(y[0,:] > np.pi))
    indy = indy.flatten()    
    w[2:nx+3, indy+2] = delta*np.cos(x[0:nx+1, indy]) + sigma/(np.cosh(sigma*(3.0*np.pi/2 - y[0:nx+1, indy])))**2
    
    w = bc(nx,ny,w)
    
    return w                
    #plt.contourf(x,y,w[1:nx+2, 1:ny+2],100,cmap='jet')
                    
#%%
# set initial condition for decay of turbulence problem
def ic_decay(nx,ny,dx,dy):
    #w = np.empty((nx+3,ny+3))
    
    epsilon = 1.0e-6
    
    kx = np.empty(nx)
    ky = np.empty(ny)
    
    kx[0:int(nx/2)] = 2*np.pi/(np.float64(nx)*dx)*np.float64(np.arange(0,int(nx/2)))
    kx[int(nx/2):nx] = 2*np.pi/(np.float64(nx)*dx)*np.float64(np.arange(-int(nx/2),0))

    ky[0:ny] = kx[0:ny]
    
    kx[0] = epsilon
    ky[0] = epsilon

    kx, ky = np.meshgrid(kx, ky, indexing='ij')
    
    ksi = 2.0*np.pi*np.random.random_sample((int(nx/2+1), int(ny/2+1)))
    eta = 2.0*np.pi*np.random.random_sample((int(nx/2+1), int(ny/2+1)))
    
    phase = np.zeros((nx,ny), dtype='complex128')
    wf =  np.empty((nx,ny), dtype='complex128')
    
    phase[1:int(nx/2),1:int(ny/2)] = np.vectorize(complex)(np.cos(ksi[1:int(nx/2),1:int(ny/2)] +
                                    eta[1:int(nx/2),1:int(ny/2)]), np.sin(ksi[1:int(nx/2),1:int(ny/2)] +
                                    eta[1:int(nx/2),1:int(ny/2)]))

    phase[nx-1:int(nx/2):-1,1:int(ny/2)] = np.vectorize(complex)(np.cos(-ksi[1:int(nx/2),1:int(ny/2)] +
                                            eta[1:int(nx/2),1:int(ny/2)]), np.sin(-ksi[1:int(nx/2),1:int(ny/2)] +
                                            eta[1:int(nx/2),1:int(ny/2)]))

    phase[1:int(nx/2),ny-1:int(ny/2):-1] = np.vectorize(complex)(np.cos(ksi[1:int(nx/2),1:int(ny/2)] -
                                           eta[1:int(nx/2),1:int(ny/2)]), np.sin(ksi[1:int(nx/2),1:int(ny/2)] -
                                           eta[1:int(nx/2),1:int(ny/2)]))

    phase[nx-1:int(nx/2):-1,ny-1:int(ny/2):-1] = np.vectorize(complex)(np.cos(-ksi[1:int(nx/2),1:int(ny/2)] -
                                                 eta[1:int(nx/2),1:int(ny/2)]), np.sin(-ksi[1:int(nx/2),1:int(ny/2)] -
                                                eta[1:int(nx/2),1:int(ny/2)]))

    k0 = 10.0
    c = 4.0/(3.0*np.sqrt(np.pi)*(k0**5))           
    
    kk = np.sqrt(kx[:,:]**2 + ky[:,:]**2)
    es = c*(kk**4)*np.exp(-(kk/k0)**2)
    wf[:,:] = np.sqrt((kk*es/np.pi)) * phase[:,:]*(nx*ny)
            
    a = pyfftw.empty_aligned((nx,ny),dtype= 'complex128')
    b = pyfftw.empty_aligned((nx,ny),dtype= 'complex128')
    
    fft_object_inv = pyfftw.FFTW(a, b,axes = (0,1), direction = 'FFTW_BACKWARD')
    ut = np.real(fft_object_inv(wf)) 
    
    #w = np.zeros((nx+3,ny+3))
    
    #periodicity
    w = np.zeros((nx+5,ny+5)) 
    w[2:nx+2,2:ny+2] = ut
    w[:,ny+2] = w[:,2]
    w[nx+2,:] = w[2,:]
    w[nx+2,ny+2] = w[2,2] 
    
    w = bc(nx,ny,w)    
    
    return w

#%%
# compute the energy spectrum numerically
def energy_spectrum(nx,ny,w):
    epsilon = 1.0e-6

    kx = np.empty(nx)
    ky = np.empty(ny)
    
    kx[0:int(nx/2)] = 2*np.pi/(np.float64(nx)*dx)*np.float64(np.arange(0,int(nx/2)))
    kx[int(nx/2):nx] = 2*np.pi/(np.float64(nx)*dx)*np.float64(np.arange(-int(nx/2),0))

    ky[0:ny] = kx[0:ny]
    
    kx[0] = epsilon
    ky[0] = epsilon

    kx, ky = np.meshgrid(kx, ky, indexing='ij')
    
    a = pyfftw.empty_aligned((nx,ny),dtype= 'complex128')
    b = pyfftw.empty_aligned((nx,ny),dtype= 'complex128')

    fft_object = pyfftw.FFTW(a, b, axes = (0,1), direction = 'FFTW_FORWARD')
    wf = fft_object(w[2:nx+2,2:ny+2]) 
    
    es =  np.empty((nx,ny))
    
    kk = np.sqrt(kx[:,:]**2 + ky[:,:]**2)
    es[:,:] = np.pi*((np.abs(wf[:,:])/(nx*ny))**2)/kk
    
    n = int(np.sqrt(nx*nx + ny*ny)/2.0)-1
    
    en = np.zeros(n+1)
    
    for k in range(1,n+1):
        en[k] = 0.0
        ic = 0
        ii,jj = np.where((kk[1:,1:]>(k-0.5)) & (kk[1:,1:]<(k+0.5)))
        ic = ii.size
        ii = ii+1
        jj = jj+1
        en[k] = np.sum(es[ii,jj])
#        for i in range(1,nx):
#            for j in range(1,ny):          
#                kk1 = np.sqrt(kx[i,j]**2 + ky[i,j]**2)
#                if ( kk1>(k-0.5) and kk1<(k+0.5) ):
#                    ic = ic+1
#                    en[k] = en[k] + es[i,j]
                    
        en[k] = en[k]/ic
        
    return en, n

#%%
def plotimage(x,y):
    fig, ax = plt.subplots(1,1,sharey=True,figsize=(6,5))
    cs1 = ax.contourf(x.T, 120, cmap = 'jet', interpolation='bilinear')
    ax.set_title("True")
    plt.colorbar(cs1, ax=ax)
    plt.show()
    
    fig, ax = plt.subplots(1,1,sharey=True,figsize=(6,5))
    cs2 = ax.contourf(y.T, 120, cmap = 'jet', interpolation='bilinear')
    ax.set_title("Coarsened")
    plt.colorbar(cs2, ax=ax)
    plt.show()
  
#%%
def export_data(nx,ny,re,n,w,s,isolver,ifm,pCU3,ifeat):
    if isolver == 1:
        if ifm == 0:
            directory = 'DNS'
            folder = f'{re:.0f}_{nx}_{ny}_seed_{seed_number}'
            folder = os.path.join(directory, folder)
        elif ifm == 1:
            directory = 'LES_DSM'
            folder = f'{re:.0f}_{nx}_{ny}'
            folder = os.path.join(directory, folder)
        elif ifm == 2:
            directory = 'LES_CNN'
            folder = f'{re:.0f}_{nx}_{ny}'
            folder = os.path.join(directory, folder)
        elif ifm == 3:
            directory = 'LES_CNN'
            folder = f'{re:.0f}_{nx}_{ny}'
            folder = os.path.join(directory, folder)
            
    elif isolver == 2:
        directory = 'Compact'
        folder = f'{re:.0f}_{nx}_{ny}'
        folder = os.path.join(directory, folder)
        
    elif isolver == 3:
        directory = 'Compact_Upwind'
        folder = f'{re:.0f}_{nx}_{ny}'
        folder = os.path.join(directory, folder)
    
    directory = os.path.join('results', folder)
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    filename = os.path.join(directory, f'field_{n}.npz')
    np.savez(filename, w=w, s=s)
         
#%% 
# assign parameters
nx = nd
ny = nd

nx_dns = n_dns
ny_dns = n_dns

pi = np.pi
lx = 2.0*pi
ly = 2.0*pi

dx = lx/np.float64(nx)
dy = ly/np.float64(ny)

ifile = 0
time = 0.0

x = np.linspace(0.0,2.0*np.pi,nx+1)
y = np.linspace(0.0,2.0*np.pi,ny+1)

x, y = np.meshgrid(x, y, indexing='ij')

#%% 
# allocate the vorticity and streamfunction arrays
w = np.empty((nx+5,ny+5)) 
s = np.empty((nx+5,ny+5))

t = np.empty((nx+5,ny+5))

r = np.empty((nx+5,ny+5))

#%%
if isolver == 1:
    if ifm == 0:
        directory = 'DNS'
        folder = f'{re:.0f}_{nx}_{ny}_seed_{seed_number}'
        folder = os.path.join(directory, folder)
    elif ifm == 1:
        directory = 'LES_DSM'
        folder = f'{re:.0f}_{nx}_{ny}'
        folder = os.path.join(directory, folder)
    elif ifm == 2:
        directory = 'LES_CNN'
        folder = f'{re:.0f}_{nx}_{ny}'
        folder = os.path.join(directory, folder)
    elif ifm == 3:
        directory = 'LES_CNN'
        folder = f'{re:.0f}_{nx}_{ny}'
        folder = os.path.join(directory, folder)
        
elif isolver == 2:
    directory = 'Compact'
    folder = f'{re:.0f}_{nx}_{ny}'
    folder = os.path.join(directory, folder)
    
elif isolver == 3:
    directory = 'Compact_Upwind'
    folder = f'{re:.0f}_{nx}_{ny}'
    folder = os.path.join(directory, folder)

directory_w = os.path.join('results', folder, 'w')
if not os.path.exists(directory_w):
    os.makedirs(directory_w)

directory_s = os.path.join('results', folder, 's')
if not os.path.exists(directory_s):
    os.makedirs(directory_s)

directory_u = os.path.join('results', folder, 'u')
if not os.path.exists(directory_u):
    os.makedirs(directory_u)

directory_v = os.path.join('results', folder, 'v')
if not os.path.exists(directory_v):
    os.makedirs(directory_v)
    
def export_data_all(w, s, u, v, n):
    if iprec == 'single':
        filename = os.path.join(directory_w, f'field_{n}.npz')
        np.savez(filename, w = np.float32(w))
        
        filename = os.path.join(directory_s, f'field_{n}.npz')
        np.savez(filename, s = np.float32(s))
        
        filename = os.path.join(directory_u, f'field_{n}.npz')
        np.savez(filename, u = np.float32(u))
        
        filename = os.path.join(directory_v, f'field_{n}.npz')
        np.savez(filename, v = np.float32(v))
        
    elif iprec == 'double':
        filename = os.path.join(directory_w, f'field_{n}.npz')
        np.savez(filename, w = w)
        
        filename = os.path.join(directory_s, f'field_{n}.npz')
        np.savez(filename, s = s)
        
        filename = os.path.join(directory_u, f'field_{n}.npz')
        np.savez(filename, u = u)
        
        filename = os.path.join(directory_v, f'field_{n}.npz')
        np.savez(filename, v = v)
    
#%%
# set the initial condition based on the problem selected
if (ipr == 1):
    w0 = ic_tgv(nx,ny,x,y)
elif (ipr == 2):
    w0 = ic_vm(nx,ny,x,y)
elif (ipr == 3):
    w0 = ic_shear(nx,ny,x,y)
elif (ipr == 4):
    if istart == 1:
        wdns = np.zeros((nx_dns+5,ny_dns+5))
        #folder_in = './results/data_'+str(int(re))+'dns_'+str(nx_dns)+'_'+str(ny_dns)
        # directory = os.path.join(ic_folder, 'results', 'DNS')
        # folder = f'{re:.0f}_{nx_dns}_{ny_dns}_seed_{seed_number}'
        # n = ic_num_snapshot
        # file_in = os.path.join(directory, folder, f'field_{n}.npz')
        directory = os.path.join('./results', 'DNS')

        # Construct the folder name using the given variables
        folder = f'{re:.0f}_{nx_dns}_{ny_dns}_seed_{seed_number}'

        # Now, add the 'name' directory and the file name
        n = ic_num_snapshot
        file_in = os.path.join(directory, folder, ic_folder,f'field_{n}.npz')

        data = np.load(file_in)
        wdns = data['w']
        w0 = np.zeros((nx+5,ny+5))
        w0[2:nx+3,2:ny+3] = coarsen(nx_dns,ny_dns,nx,ny,wdns[2:nx_dns+3,2:ny_dns+3])
        w0 = bc(nx,ny,w0)
    else:
        w0 = ic_decay(nx,ny,dx,dy)
    
w = np.copy(w0)
s = fps(nx, ny, dx, dy, -w)
s = bc(nx,ny,s)

sx,sy = grad_spectral(nx,ny,s[2:nx+3,2:ny+3])
u, v = sy, -sx

n = 0
export_data_all(w, s, u, v, n)

#%%

def rhs(nx,ny,dx,dy,re,pCU3,w,s,max_min,model,ifm,isolver,ifeat):
    if isolver == 1:
        return rhs_arakawa(nx,ny,dx,dy,re,w,s,ifm,kappa,max_min,model,ifeat)
    if isolver == 2:
        return rhs_compactv(nx,ny,dx,dy,re,w,s)
    if isolver == 3:
        return rhs_cu3v(nx,ny,dx,dy,re,pCU3,w,s)
   

#%%
# time integration using third-order Runge Kutta method
aa = 1.0/3.0
bb = 2.0/3.0
clock_time_init = tm.time()

for k in range(1,nt+1):
    time = time + dt
    
    r = rhs(nx,ny,dx,dy,re,pCU3,w,s,max_min,model,ifm,isolver,ifeat)
    
    #stage-1
    t[2:nx+3,2:ny+3] = w[2:nx+3,2:ny+3] + dt*r[2:nx+3,2:ny+3]
    
    t = bc(nx,ny,t)
    
    s = fps(nx, ny, dx, dy, -t)
    s = bc(nx,ny,s)
    
    r = rhs(nx,ny,dx,dy,re,pCU3,t,s,max_min,model,ifm,isolver,ifeat)

    #stage-2
    t[2:nx+3,2:ny+3] = 0.75*w[2:nx+3,2:ny+3] + 0.25*t[2:nx+3,2:ny+3] + 0.25*dt*r[2:nx+3,2:ny+3]
    
    t = bc(nx,ny,t)
    
    s = fps(nx, ny, dx, dy, -t)
    s = bc(nx,ny,s)
    
    r = rhs(nx,ny,dx,dy,re,pCU3,t,s,max_min,model,ifm,isolver,ifeat)

    #stage-3
    w[2:nx+3,2:ny+3] = aa*w[2:nx+3,2:ny+3] + bb*t[2:nx+3,2:ny+3] + bb*dt*r[2:nx+3,2:ny+3]
    
    w = bc(nx,ny,w)
    
    s = fps(nx, ny, dx, dy, -w)
    s = bc(nx,ny,s)
    
    if (k % freq == 0):
        print('%0.4i %0.3f %0.3f %0.3f' % (k, time, np.max(w), np.min(w)))
        n = int(k/freq)
              
        sx,sy = grad_spectral(nx,ny,s[2:nx+3,2:ny+3])
        u, v = sy, -sx
        
        export_data_all(w, s, u, v, n)
        

total_clock_time = tm.time() - clock_time_init
print(f'Total wall time = {total_clock_time:0.3f}')

# filename = os.path.join(directory, 'cpu_time.txt')
# fo = open(filename, "w")
# fo.write(f'Total wall time = {total_clock_time:0.3f}')
# fo.close()

#%%
# exact solution for TGV problem
if (ipr == 1):
    we = exact_tgv(nx,ny,x,y,time,re)
    
# compute the exact, initial and final energy spectrum
if (ipr == 4):       
    ent, n = energy_spectrum(nx,ny,w)
    en0, n = energy_spectrum(nx,ny,w0)
    kw = np.linspace(1,n,n)
    
    k0 = 10.0
    c = 4.0/(3.0*np.sqrt(np.pi)*(k0**5))           
    ese = c*(kw**4)*np.exp(-(kw/k0)**2)
    
    fig, ax = plt.subplots(1,1,figsize=(6,6))
    
    line = 100*kw**(-3.0)
    
    ax.loglog(kw,ese[:],'k', lw = 2, label='Exact')
    ax.loglog(kw,en0[1:],'r', ls = '-', lw = 2, label=f'$t = {ic_num_snapshot*dt:0.1f}$')
    ax.loglog(kw,ent[1:], 'b', lw = 2, label = f'$t = {dt*nt + + ic_num_snapshot*dt:0.1f}$')
    ax.loglog(kw,line, 'g--', lw = 2, label = '$k^{-3}$')
    
    ax.set_xlabel('$K$')
    ax.set_ylabel('$E(K)$')
    ax.legend(loc=0)
    ax.set_ylim([1e-8,1e0])  
    ax.set_xlim([1e0,1e3])  
    filename = os.path.join(directory, f'energy_spectrum_{re}_{seed_number}.png')
    fig.savefig(filename, bbox_inches = 'tight', dpi=100)
    
#%%
# contour plot for initial and final vorticity
fig, axs = plt.subplots(1,2,figsize=(9,5))

vmin = np.int(np.min(w0))
vmax = np.int(np.max(w0))
cs = axs[0].contourf(x,y,w0[2:nx+3,2:ny+3], 20, vmin=vmin, vmax=vmax, cmap = 'jet')
axs[0].set_aspect('equal')
fig.colorbar(cs, ax=axs[0], shrink=0.7)
axs[0].set_title(f'$t$ = {ic_num_snapshot*dt:0.1f}')

vmin = np.int(np.min(w))
vmax = np.int(np.max(w))
cs = axs[1].contourf(x,y,w[2:nx+3,2:ny+3], 20, vmin=vmin, vmax=vmax, cmap = 'jet')
axs[1].set_aspect('equal')
fig.colorbar(cs, ax=axs[1], shrink=0.7)
axs[1].set_title(f'$t = {dt*nt + ic_num_snapshot*dt:0.1f}$')

axs[0].set_xlabel('$x$')
axs[0].set_ylabel('$y$')
axs[1].set_xlabel('$x$')
axs[1].set_ylabel('$y$')

fig.tight_layout() 
plt.show()

filename = os.path.join(directory, f'field_{re}_{seed_number}.png')
fig.savefig(filename, bbox_inches = 'tight', dpi=100)



    



