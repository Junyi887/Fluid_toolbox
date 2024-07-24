
import numpy as np
import dedalus.public as d3
import logging

import argparse 
parser = argparse.ArgumentParser(description='Burgers 2d solver')
parser.add_argument('--seed', type=int, default=344)

args = parser.parse_args()
print(args)
logger = logging.getLogger(__name__)

def initialize_field(x_range, y_range, grid_size, order=4, seed=0):
    """
    Initialize the field based on a truncated Fourier series with random coefficients.
    
    :param x_range: tuple, range of x values e.g., (0, 1)
    :param y_range: tuple, range of y values e.g., (0, 1)
    :param grid_size: int, size of the grid
    :param order: int, order of the Fourier series
    :param seed: int, random seed for reproducibility
    
    :return: np.array, initialized field
    """
    np.random.seed(seed)
    
    # Create a grid of x and y values
    x = np.linspace(x_range[0], x_range[1], grid_size)
    y = np.linspace(y_range[0], y_range[1], grid_size)
    x, y = np.meshgrid(x, y)
    
    # Generate random coefficients
    lam = np.random.randn(2, 2, (2 * order + 1) ** 2)
    c = -1 + 2 * np.random.rand(2)
    
    # Compute the field values based on the Fourier series
    w_sin = np.zeros_like(x)
    w_cos = np.zeros_like(x)
    for i in range(-order, order + 1):
        for j in range(-order, order + 1):
            idx = (i + order) * (2 * order + 1) + (j + order)   # Flatten index
            w_sin += lam[0, 0, idx] * np.sin(2 * np.pi * (i * x + j * y))
            w_cos += lam[1, 0, idx] * np.cos(2 * np.pi * (i * x + j * y))
    
    # Normalize and add the constant vector c
    u_sin = 2 * w_sin / np.max(np.abs(w_sin)) + c[0]
    u_cos = 2 * w_cos / np.max(np.abs(w_cos)) + c[1]
    
    return np.stack((u_sin, u_cos),axis=0)  # Combine the fields
# Parameters
Lx, Lz = 1, 1
Nx, Nz = 128, 128
Rayleigh = 100
dealias = 3/2
stop_sim_time = 2
timestepper = d3.RK222
max_timestep = 1e-4
dtype = np.float32
nu = 1/Rayleigh
# Bases
coords = d3.CartesianCoordinates('x', 'z')
dist = d3.Distributor(coords, dtype=dtype)
xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(0, Lx), dealias=dealias)
zbasis = d3.RealFourier(coords['z'], size=Nz, bounds=(0, Lz), dealias=dealias)

# Fields
u = dist.VectorField(coords, name='u', bases=(xbasis,zbasis))

# Substitutions

x, z = dist.local_grids(xbasis, zbasis)
ex, ez = coords.unit_vector_fields(dist)

# Problem
# First-order form: "div(f)" becomes "trace(grad_f)"
# First-order form: "lap(f)" becomes "div(grad_f)"
problem = d3.IVP([u], namespace=locals())
problem.add_equation("dt(u) - nu*lap(u) = - u@grad(u)")

# Solver
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time

# Initial conditions
uv0 = initialize_field((0, Lx), (0, Lz), Nx, order=4, seed=args.seed)
u['g'] = uv0


# Analysis
snapshots = solver.evaluator.add_file_handler(f"../burger2D_diff_IC/Burger2D_{args.seed}", sim_dt=0.001, max_writes=2000)
snapshots.add_task(-d3.div(d3.skew(u)), name='vorticity')
snapshots.add_task(u@ez, name='v')
snapshots.add_task(u@ex, name='u')
# CFL
CFL = d3.CFL(solver, initial_dt=max_timestep, cadence=10, safety=0.5, threshold=0.05,
             max_change=1.5, min_change=0.5, max_dt=max_timestep)
CFL.add_velocity(u)

# Flow properties
flow = d3.GlobalFlowProperty(solver, cadence=10)
flow.add_property(np.sqrt(u@u)/nu, name='Re')

# Main loop
startup_iter = 100
try:
    logger.info('Starting main loop')
    while solver.proceed:
        timestep = max_timestep
        solver.step(timestep)
        if (solver.iteration-1) % 10 == 0:
            max_Re = flow.max('Re')
            logger.info('Iteration=%i, Time=%e, dt=%e, max(Re)=%f' %(solver.iteration, solver.sim_time, timestep, max_Re))
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()