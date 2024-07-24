import matplotlib.pyplot as plt
import numpy as np
import seaborn
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import seaborn as sns
import xarray
import numpy as np
import jax_cfd.base as cfd
import jax_cfd.base.grids as grids
import jax_cfd.spectral as spectral

import dataclasses
import argparse
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--res', type=int, default=1024)
parser.add_argument('--scale', type=int, default=4 )
parser.add_argument('--seed', type=int, default=1220)
args=parser.parse_args()

# physical parameters (from Li et al. 2020)
viscosity = 0.001 # !! Re = 500 viscosity is scaled by the maximum force amplitude, not simply \nu = 1/Re
max_velocity = 7
psedo_final_time = 20
FINAL_SIM_TIME = 5
SAVED_SNAPSHOTS = 500

resol = args.res
scale = args.scale
seed = args.seed

grid_hr = grids.Grid((resol, resol), domain=((0, 2 * jnp.pi), (0, 2 * jnp.pi)))
grid_lr = grids.Grid((resol//scale, resol//scale), domain=((0, 2 * jnp.pi), (0, 2 * jnp.pi)))

dt = cfd.equations.stable_time_step(max_velocity, .5, viscosity, grid_hr)
print(f"dt from CFL condition based on HR grid is {dt}")

# setup step function using crank-nicolson runge-kutta order 4
smooth = True # use anti-aliasing 

step_fn = spectral.time_stepping.crank_nicolson_rk4(
    spectral.equations.ForcedNavierStokes2D(viscosity, grid_hr, smooth=smooth), dt)

# run the simulation up until time 20 to get real intial conditions
final_time = psedo_final_time 
outer_steps = SAVED_SNAPSHOTS
inner_steps = (final_time // dt) // outer_steps

import time
start = time.time()

trajectory_fn = cfd.funcutils.trajectory(
    cfd.funcutils.repeated(step_fn, inner_steps), outer_steps)
v0 = cfd.initial_conditions.filtered_velocity_field(jax.random.PRNGKey(seed), grid_hr, max_velocity, 4)
vorticity0 = cfd.finite_differences.curl_2d(v0).data
print(f"vorticitiy0 shape: {vorticity0.shape}, type {type(vorticity0)}")
vorticity_hat0 = jnp.fft.rfftn(vorticity0)
_, trajectory = trajectory_fn(vorticity_hat0)
spatial_coord = jnp.arange(grid_hr.shape[0]) * 2 * jnp.pi / grid_hr.shape[0] # same for x and y
coords = {
  'time': dt * jnp.arange(outer_steps) * inner_steps,
  'x': spatial_coord,
  'y': spatial_coord,
}
w_sol =jnp.fft.irfftn(trajectory, axes=(1,2))

print(w_sol.shape)
time_stemp = time.time()
print(f"time elapsed for generating initial conditon is {time_stemp - start} s")

ic_hr = w_sol[-1]
ic_lr = ic_hr[::scale, ::scale]

ic_hr_hat = jnp.fft.rfftn(ic_hr)
ic_lr_hat = jnp.fft.rfftn(ic_lr)

final_time = FINAL_SIM_TIME 
outer_steps = 500
inner_steps = (final_time // dt) // outer_steps

step_fn_hr = spectral.time_stepping.crank_nicolson_rk4(
    spectral.equations.ForcedNavierStokes2D(viscosity, grid_hr, smooth=smooth), dt)
step_fn_lr = spectral.time_stepping.crank_nicolson_rk4(
    spectral.equations.ForcedNavierStokes2D(viscosity, grid_lr, smooth=smooth), dt)

trajectory_fn_HR = cfd.funcutils.trajectory(
    cfd.funcutils.repeated(step_fn_hr, inner_steps), outer_steps)
trajectory_fn_LR = cfd.funcutils.trajectory(
    cfd.funcutils.repeated(step_fn_lr, inner_steps), outer_steps)

_, trajectory_hr = trajectory_fn_HR(ic_hr_hat)
_, trajectory_lr = trajectory_fn_LR(ic_lr_hat)

# transform the trajectory into real-space and wrap in xarray for plotting


spatial_coord_HR = jnp.arange(grid_hr.shape[0]) * 2 * jnp.pi / grid_hr.shape[0] # same for x and y
spatial_coord_LR = jnp.arange(grid_lr.shape[0]) * 2 * jnp.pi / grid_lr.shape[0] # same for x and y

to_velocity_hr = spectral.utils.vorticity_to_velocity(grid_hr)
to_velocity_lr = spectral.utils.vorticity_to_velocity(grid_lr)

uv_hr = to_velocity_hr(trajectory_hr)
uv_lr = to_velocity_lr(trajectory_lr)

t = 10 + dt*jnp.arange(outer_steps) * inner_steps
w_hr = jnp.fft.irfftn(trajectory_hr, axes=(1,2))
w_lr = jnp.fft.irfftn(trajectory_lr, axes=(1,2))
u_hr = jnp.fft.irfftn(uv_hr[0], axes=(1,2))
v_hr = jnp.fft.irfftn(uv_hr[1], axes=(1,2))
u_lr = jnp.fft.irfftn(uv_lr[0], axes=(1,2))
v_lr = jnp.fft.irfftn(uv_lr[1], axes=(1,2))
# time 
time_hr_lr = time.time() 
print(f"time elapsed for generating hr/lr pairs is {time_hr_lr- time_stemp} s")

'''load to h5 file'''
import h5py # /pscratch/sd/j/junyi012/jax_cfd_DNS/
with h5py.File(f'/pscratch/sd/j/junyi012/NS_lrsim_s4/NS_lres_sim_s{scale}_{seed}.h5', 'w') as f:
    tasks = f.create_group('tasks')
    tasks.create_dataset('u', data=u_hr.astype(jnp.float32))
    tasks.create_dataset('v', data=v_hr.astype(jnp.float32))
    tasks.create_dataset('vorticity', data=w_hr.astype(jnp.float32))
    tasks.create_dataset('t', data=t.astype(jnp.float32))
    tasks.create_dataset('u_lr', data=u_lr.astype(jnp.float32))
    tasks.create_dataset('v_lr', data=v_lr.astype(jnp.float32))
    tasks.create_dataset('vorticity_lr', data=w_lr.astype(jnp.float32))
time_save = time.time() 
print(f"time elapsed for saving is {time_save - time_hr_lr} s")

print(f"total time elapsed is {time.time() - start} s")
print(f"data summary: resol {resol}, scale {scale} \n seed {seed} \n saved_dt {t[2]-t[1]}, sim_dt {dt},saved_snapshots {SAVED_SNAPSHOTS}, Time range [0, {FINAL_SIM_TIME}]")

# plot the correlation between HR and LR
# fig, axs = plt.subplots(5, 5, figsize=(8, 8))
# i = 0
# for ax in axs:
#     for a in ax:
#         a.set_axis_off()
#         a.imshow(w_hr[-26+i], cmap='RdBu_r')
#         i +=1 
# fig.savefig(figures)