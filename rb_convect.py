"""
Author: Tom Joshi-Cale
"""
# ===================
# ======IMPORTS======
# ===================
import numpy as np
import logging
import os
from datetime import datetime
import time

import argparse
import run_params as rp

import dedalus.public as de
from dedalus.extras import flow_tools

logger = logging.getLogger(__name__)


# ====================
# =====CLA PARSING====
# ====================
parser = argparse.ArgumentParser()
parser.add_argument("-t", "--test", help="Do not save any output", action="store_true")

parser.add_argument(
    "-o",
    "--output_folder",
    help="Name of folder to store output data in. Default=output/",
    default="output/",
)

parser.add_argument("-i", "--initial", help="Path to file to read input from.")

args = parser.parse_args()
if args.test:
    save = False
else:
    save = True

if save:
    outpath = os.path.normpath(args.output_folder) + "/"
    os.makedirs(outpath, exist_ok=True)

# ====================
# FUNCTION DEFINITIONS
# ====================
def initialise_problem(domain, Ra, Pr):
    problem = de.IVP(domain, variables=["u", "w", "P", "T", "uz", "wz", "Tz"])
    problem.parameters["Ra"] = Ra
    problem.parameters["Pr"] = Pr
    problem.parameters["L"] = a
    problem.parameters["D"] = 1

    # Set up d/dz equations
    problem.add_equation("Tz - dz(T) = 0")  # Allows Tz as shorthand for dT/dz
    problem.add_equation("uz - dz(u) = 0")  # Allows uz as shorthand for du/dz
    problem.add_equation("wz - dz(w) = 0")  # Allows wz as shorthand for dw/dz

    # Mass continuity equation
    problem.add_equation("dx(u) + wz = 0")

    # x-component of Navier Stokes equation
    problem.add_equation("dt(u) - Pr*(dx(dx(u)) + dz(uz)) + dx(P) = -(u*dx(u) + w*uz)")

    # z-component of Navier Stokes equation
    problem.add_equation(
        "dt(w) - Pr*(dx(dx(w)) + dz(wz)) - Ra*Pr*T + dz(P) = -(u*dx(w) + w*wz)"
    )

    # Temperature equation
    problem.add_equation("dt(T) - (dx(dx(T)) + dz(Tz)) = -(u*dx(T) + w*Tz)")

    # ====================
    # Add boundary conditions
    # ====================
    # Stress-Free horizontal boundaries
    problem.add_bc("left(uz) = 0")
    problem.add_bc("right(uz) = 0")

    # Impermeable side boundaries)
    problem.add_bc("left(w) = 0")
    problem.add_bc("right(w) = 0", condition="(nx != 0)")
    problem.add_bc("right(P) = 0", condition="(nx == 0)")
    #
    # Top boundary fixed at T=0
    problem.add_bc("right(T) = 0")
    # No heat flow through left boundary
    problem.add_bc("left(Tz) = -1")

    return problem


def analysis_task_setup(solver, outpath, an_iter):
    analysis = solver.evaluator.add_file_handler(
        outpath + "analysis", iter=an_iter, max_writes=5000
    )

    # Conductive Heat Flux
    analysis.add_task("integ( (-1)*Tz, 'x')/L", layout="g", name="L_cond")

    # Convective Heat Flux
    analysis.add_task("integ( T * w, 'x') * Pr / L", layout="g", name="L_conv")

    # Kinetic Energy
    analysis.add_task(
        "integ( integ( 0.5*(u*u + w*w), 'x'), 'z')/D", layout="g", name="KE"
    )

    return analysis


# ====================
# Initialisation
# ====================

if not args.initial:
    a = rp.a
    Nx, Nz = rp.Nx, rp.Nz
    Pr = rp.Pr
    Ra = rp.Ra
else:
    print("Reading initial conditions not yet implemented")

# ====================
# Create basis and domain
# ====================
xbasis = de.Fourier("x", Nx, interval=(0, a), dealias=3 / 2)
zbasis = de.Chebyshev("z", Nz, interval=(0, 1), dealias=3 / 2)
domain = de.Domain([xbasis, zbasis], grid_dtype=np.float64)

# ====================
# Set Up Problems
# ====================
problem = initialise_problem(domain, Ra, Pr)

# ====================
# Build IVP Solver
# ====================
solver = problem.build_solver(de.timesteppers.RK222)
logger.info("Solver built")
print("=============\n")

# ====================
# Initial Conditions
# ====================
if not args.initial:
    z = domain.grid(1)
    T = solver.state["T"]
    Tz = solver.state["Tz"]
    # Random temperature perturbations
    gshape = domain.dist.grid_layout.global_shape(scales=1)
    slices = domain.dist.grid_layout.slices(scales=1)
    rand = np.random.RandomState(seed=42)
    noise = rand.standard_normal(gshape)[slices]

    # Linear background + damped perturbations at walls
    zb, zt = zbasis.interval
    pert = 1e-5 * noise * (zt - z) * (z - zb)
    T["g"] = pert
    T.differentiate("z", out=Tz)

    dt = rp.dt

    fh_mode = "overwrite"
else:
    print("initial condition reading not yet implemented")
    exit(-999)
    write, last_dt = solver.load_state("restart.h5", -1)

    dt = last_dt

    fh_mode = "append"

max_dt = rp.max_dt
solver.stop_sim_time = rp.end_sim_time
solver.stop_wall_time = rp.end_wall_time
solver.stop_iteration = rp.end_iteration
# ====================
# CFL Conditions
# ====================
cfl = flow_tools.CFL(
    solver,
    initial_dt=dt,
    cadence=10,
    safety=0.5,
    max_change=1.5,
    min_change=0.5,
    max_dt=max_dt,
    threshold=0.05,
)
cfl.add_velocities(("u", "w"))

# Flow properties
flow = flow_tools.GlobalFlowProperty(solver, cadence=10)
flow.add_property("sqrt(u*u + w*w)", name="Re")

# Save snapshots
snapshots = solver.evaluator.add_file_handler(
    outpath + "snapshots", iter=rp.snapshot_iter, max_writes=5000
)
snapshots.add_system(solver.state)

# Analysis tasks
analysis = analysis_task_setup(solver, outpath, rp.analysis_iter)


try:
    logger.info("Starting loop")
    start_time = time.time()
    while solver.ok:
        dt = cfl.compute_dt()
        dt = solver.step(dt)
        if (solver.iteration - 1) % 10 == 0:
            logger.info(
                "Iteration {}, Time: {:1.3e}, dt: {:1.3e}".format(
                    solver.iteration, solver.sim_time, dt
                )
            )
            logger.info("Max Re = {:1.3e}".format(flow.max("Re")))
except:
    logger.error("Exception raised, triggering end of main loop.")
    raise
finally:
    end_time = time.time()
    logger.info("Iterations: {}".format(solver.iteration))
    logger.info("Sim end time: {:1.3e}".format(solver.sim_time))
    logger.info("Run time: {:.3f}s".format(time.time() - start_time))
    logger.info(
        "Run time: {:1.3e} cpu-hr".format(
            (end_time - start_time) / 60 / 60 * domain.dist.comm_cart.size
        )
    )
