import jax
import jax.numpy as jnp
import jax_cfd.base as cfd
from jax_cfd.base import grids
from jax_cfd.base import boundaries
from jax_cfd.base import funcutils
from jax_cfd.base import pressure
import numpy as np
import seaborn
import xarray
import matplotlib.pyplot as plt
# check old data
import h5py

# f = h5py.File('/pscratch/sd/j/junyi012/Decay_Turbulence_small/train/Decay_turb_small_128x128_125.h5', 'r')
# w_old = f["tasks"]["vorticity"][()]
# print(f["tasks"].keys())
# print("old data",w_old.shape)
import argparse
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--res', type=int, default=128)
parser.add_argument('--scale', type=int, default=4)
parser.add_argument('--seed', type=int, default=110)
args=parser.parse_args()

size = args.res
scale = args.scale
density = 1.
viscosity = 1e-3
seed = args.seed
outer_steps = 200
final_time = 10 
#dt = 0.001
max_velocity = 2.0
cfl_safety_factor = 0.5

# Define the physical dimensions of the simulation.
grid_hr = cfd.grids.Grid((size, size), domain=((0, 2 * jnp.pi), (0, 2 * jnp.pi)))
grid_lr = cfd.grids.Grid((size//scale, size//scale), domain=((0, 2 * jnp.pi), (0, 2 * jnp.pi)))
# Construct a random initial velocity. The `filtered_velocity_field` function
# ensures that the initial velocity is divergence free and it filters out
# high frequency fluctuations.
import os
if os.path.exists(f'data_gen/DT_IC/initial_velocity_{seed}_u.npy') and os.path.exists(f'data_gen/DT_IC/initial_velocity_{seed}_v.npy'):
    u_hr = np.load(f'data_gen/DT_IC/initial_velocity_{seed}_u.npy')
    v_hr = np.load(f'data_gen/DT_IC/initial_velocity_{seed}_v.npy')
    print("load initial velocity from file")
else:
    print(f"generate initial velocity for seed {seed}")
    v0_hr = cfd.initial_conditions.filtered_velocity_field(
        jax.random.PRNGKey(seed), grid_hr, max_velocity)
    u_hr = v0_hr[0].data
    v_hr = v0_hr[1].data
    np.save(f'data_gen/DT_IC/initial_velocity_{seed}_u.npy',u_hr)
    np.save(f'data_gen/DT_IC/initial_velocity_{seed}_v.npy',v_hr)


u_lr = u_hr[::scale,::scale]
v_lr = v_hr[::scale,::scale]


def load_lr_IC(u_lr,v_lr,iterations: int = 3, maximum_velocity: float = max_velocity):
    GridVariable = grids.GridVariable
    GridVariableVector = grids.GridVariableVector
    def _max_speed(v):
        return jnp.linalg.norm(jnp.array([u.data for u in v]), axis=0).max()
    boundary_conditions = []
    velocity_components = []
    velocity_components.append(u_lr)
    velocity_components.append(v_lr)
    boundary_conditions.append(boundaries.periodic_boundary_conditions(grid_lr.ndim))
    boundary_conditions.append(boundaries.periodic_boundary_conditions(grid_lr.ndim))
    velocity= cfd.initial_conditions.wrap_variables(velocity_components, grid_lr, boundary_conditions)
    def project_and_normalize(v: GridVariableVector):
        v = pressure.projection(v)
        vmax = _max_speed(v)
        v = tuple(
            grids.GridVariable(maximum_velocity * u.array / vmax, u.bc) for u in v)
        return v
  # Due to numerical precision issues, we repeatedly normalize and project the
  # velocity field. This ensures that it is divergence-free and achieves the
  # specified maximum velocity.
    return funcutils.repeated(project_and_normalize, iterations)(velocity)

v0_lr = load_lr_IC(u_lr,v_lr)

fig,ax = plt.subplots(2,2,figsize=(5,5))
ax[0,0].imshow(v0_hr[0].data)
ax[0,1].imshow(v0_lr[0].data)
ax[1,0].imshow(v0_hr[1].data)
ax[1,1].imshow(v0_lr[1].data)
fig.savefig(f'data_gen/initial_velocity_{seed}.png')

# Choose a time step
dt_lr = cfd.equations.stable_time_step(
    max_velocity, cfl_safety_factor, viscosity, grid_lr)
print("cfl dt lr = ", dt_lr)
dt_hr = cfd.equations.stable_time_step(
    max_velocity, cfl_safety_factor, viscosity, grid_hr)
print("cfl dt hr = ", dt_hr)

# Define a step function and use it to compute a trajectory.
step_fn_lr = cfd.funcutils.repeated(
    cfd.equations.semi_implicit_navier_stokes(
        density=density, viscosity=viscosity, dt=dt_hr, grid=grid_lr),
    steps=inner_steps)
rollout_fn_lr = jax.jit(cfd.funcutils.trajectory(step_fn_lr, outer_steps))
time,trajectory_lr = jax.device_get(rollout_fn_lr(v0_lr))
print(f"*******{time} in lr")
dt = 0.001
step_fn_hr = cfd.funcutils.repeated(
    cfd.equations.semi_implicit_navier_stokes(
        density=density, viscosity=viscosity, dt=dt_hr, grid=grid_hr),
    steps=inner_steps)
rollout_fn_hr = jax.jit(cfd.funcutils.trajectory(step_fn_hr, outer_steps))
time,trajectory_hr = jax.device_get(rollout_fn_hr(v0_hr))
print(f"*******{time} in hr")
# load into xarray for visualization and analysis
ds_lr = xarray.Dataset(
    {
        'u': (('time', 'x', 'y'), trajectory_lr[0].data),
        'v': (('time', 'x', 'y'), trajectory_lr[1].data),
    },
    coords={
        'x': grid_lr.axes()[0],
        'y': grid_lr.axes()[1],
        'time': dt * inner_steps * np.arange(outer_steps)
    }
)    
ds_hr = xarray.Dataset(
    {
        'u': (('time', 'x', 'y'), trajectory_hr[0].data),
        'v': (('time', 'x', 'y'), trajectory_hr[1].data),
    },
    coords={
        'x': grid_hr.axes()[0],
        'y': grid_hr.axes()[1],
        'time': dt * inner_steps * np.arange(outer_steps)
    }
)    

# def vorticity(ds):
#   return (ds.v.differentiate('x') - ds.u.differentiate('y')).rename('vorticity')

# ds_lr['vorticity'] = vorticity(ds_lr)
# ds_hr['vorticity'] = vorticity(ds_hr)
# # (ds_lr.pipe(vorticity).thin(time=20)
# #  .plot.imshow(col='time', cmap=seaborn.cm.icefire, robust=True, col_wrap=5))
# '''load to h5 file'''
# import h5py
# u_lr = ds_lr['u'].values
# v_lr = ds_lr['v'].values
# w_lr = ds_lr['vorticity'].values
# u_hr = ds_hr['u'].values
# v_hr = ds_hr['v'].values
# w_hr = ds_hr['vorticity'].values
# t = ds_lr['time'].values
# print(f"u_lr.shape = {u_lr.shape}, v_lr.shape = {v_lr.shape}, vorticity_lr.shape = {w_lr.shape}")
# print(f"u_hr.shape = {u_hr.shape}, v_hr.shape = {v_hr.shape}, vorticity_hr.shape = {w_hr.shape}")
# print(f"type of u_lr = {type(u_lr)}, type of v_lr = {type(v_lr)}, type of vorticity_lr = {type(w_lr)}")
# print(f"type of u_hr = {type(u_hr)}, type of v_hr = {type(v_hr)}, type of vorticity_hr = {type(w_hr)}")
# # trim data
# u_hr = u_hr[50:550].astype(jnp.float32)
# v_hr = v_hr[50:550].astype(jnp.float32)
# w_hr = w_hr[50:550].astype(jnp.float32)
# u_lr = u_lr[50:550].astype(jnp.float32)
# v_lr = v_lr[50:550].astype(jnp.float32)
# w_lr = w_lr[50:550].astype(jnp.float32)
# t = t[50:550]  
# with h5py.File(f'decay_turb_lres_sim_res{size}_s{scale}_{seed}.h5', 'w') as f:
#     tasks = f.create_group('tasks')
#     tasks.create_dataset('u', data=u_hr)
#     tasks.create_dataset('v', data=v_hr)
#     tasks.create_dataset('vorticity', data=w_hr)
#     tasks.create_dataset('t', data=t)
#     tasks.create_dataset('u_lr', data=u_lr)
#     tasks.create_dataset('v_lr', data=v_lr)
#     tasks.create_dataset('vorticity_lr', data=w_lr)

# print(t)
import matplotlib.pyplot as plt     

# corrlation 
# baseline_palette = seaborn.color_palette('YlGnBu', n_colors=7)[1:]
# correlation = summary.vorticity_correlation.sel(time=slice(20)).compute()