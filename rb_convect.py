"""
2D Boussinesq Rayleigh-Bénard Convection

Equations are for the y-z plane, and have been non-dimensionalised using the
viscous time.

Author: Tom Joshi-Cale

TO DO:

"""
# ===================
# ======IMPORTS======
# ===================
import numpy as np
import logging
import os
from datetime import datetime
import time
import pathlib
import shutil

import argparse
import run_params as rp

import dedalus.public as de
from dedalus.extras import flow_tools

logger = logging.getLogger(__name__)


class NaNFlowError(Exception):
    pass


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

parser.add_argument("-i", "--initial", help="Path to folder to read input from.")

args = parser.parse_args()
if args.test:
    save = False
else:
    save = True

if save:
    outpath = os.path.normpath(args.output_folder) + "/"
    os.makedirs(outpath, exist_ok=True)

if args.initial:
    restart_path = os.path.normpath(args.initial) + "/"

# ====================
# FUNCTION DEFINITIONS
# ====================
def initialise_problem(domain, Ra, Pr):
    problem = de.IVP(domain, variables=["v", "w", "P", "T", "vz", "wz", "Tz"])
    problem.parameters["Ra"] = Ra
    problem.parameters["Pr"] = Pr
    problem.parameters["L"] = a
    problem.parameters["D"] = 1

    # Set up d/dz equations
    problem.add_equation("Tz - dz(T) = 0")  # Allows Tz as shorthand for dT/dz
    problem.add_equation("vz - dz(v) = 0")  # Allows vz as shorthand for dv/dz
    problem.add_equation("wz - dz(w) = 0")  # Allows wz as shorthand for dw/dz

    # Mass continuity equation
    problem.add_equation("dy(v) + wz = 0")

    # y-component of Navier Stokes equation
    problem.add_equation("dt(v) - (dy(dy(v)) + dz(vz)) + dy(P) = -(v*dy(v) + w*vz)")

    # z-component of Navier Stokes equation
    problem.add_equation(
        "dt(w) - (dy(dy(w)) + dz(wz)) - (Ra/Pr)*T + dz(P) = -(v*dy(w) + w*wz)"
    )

    # Temperature equation
    problem.add_equation("dt(T) - (1/Pr)*(dy(dy(T)) + dz(Tz)) = -(v*dy(T) + w*Tz)")

    # ====================
    # Add boundary conditions
    # ====================
    # Stress-Free horizontal boundaries
    problem.add_bc("left(vz) = 0")
    problem.add_bc("right(vz) = 0")

    # Impermeable side boundaries)
    problem.add_bc("left(w) = 0")
    problem.add_bc("right(w) = 0", condition="(ny != 0)")
    problem.add_bc("right(P) = 0", condition="(ny == 0)")
    #
    # Top boundary fixed at T=0
    problem.add_bc("right(T) = 0")
    # Fixed heat flux through left boundary
    problem.add_bc("left(Tz) = -1")

    return problem


def analysis_task_setup(solver, outpath, an_iter):
    analysis = solver.evaluator.add_file_handler(
        outpath + "analysis", iter=an_iter, max_writes=5000
    )

    # Conductive Heat Flux
    analysis.add_task("integ( (-1)*Tz, 'y')/L", layout="g", name="L_cond")

    # Convective Heat Flux
    analysis.add_task("integ( T * w, 'y') * Pr / L", layout="g", name="L_conv")

    # Kinetic Energy
    analysis.add_task(
        "integ( integ( 0.5*(v*v + w*w), 'y'), 'z')/D", layout="g", name="KE"
    )

    return analysis


# ====================
# Initialisation
# ====================

if not args.initial:
    a = rp.a
    Ny, Nz = rp.Ny, rp.Nz
    Pr = rp.Pr
    Ra = rp.Ra
else:
    with h5py.File(restart_path + "run_params/run_params_s1.h5", "r") as f:
        a = int(np.array(f["tasks"]["a"]))
        Ny = int(np.array(f["tasks"]["Ny"]))
        Nz = int(np.array(f["tasks"]["Nz"]))
        Pr = float(np.array(f["tasks"]["Pr"]))
        Ra = float(np.array(f["tasks"]["Ra"]))

# ====================
# Create basis and domain
# ====================
ybasis = de.Fourier("y", Ny, interval=(0, a), dealias=3 / 2)
zbasis = de.Chebyshev("z", Nz, interval=(0, 1), dealias=3 / 2)
domain = de.Domain([ybasis, zbasis], grid_dtype=np.float64)

# ====================
# Set Up Problems
# ====================
problem = initialise_problem(domain, Ra, Pr)

# ====================
# Build IVP Solver
# ====================
solver = problem.build_solver(de.timesteppers.RK222)
logger.info("Solver built")

# ====================
# Initial Conditions
# ====================
if not args.initial:
    y = domain.grid(0)
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
    first_iter = 0

    dt = rp.dt

    fh_mode = "overwrite"
else:
    if pathlib.Path(restart_path + "snapshots/snapshots_s1.h5").exists():
        write, last_dt = solver.load_state(restart_path + "snapshots_s1.h5", -1)
    else:
        print("{}restart.h5 does not exist.".format(restart_path + "snapshots_s1.h5"))
        exit(-10)

    dt = last_dt
    first_iter = solver.iteration

    fh_mode = "append"

max_dt = rp.max_dt
solver.stop_sim_time = rp.end_sim_time
solver.stop_wall_time = rp.end_wall_time
solver.stop_iteration = first_iter + rp.end_iteration + 1
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
cfl.add_velocities(("v", "w"))

# Flow properties
flow = flow_tools.GlobalFlowProperty(solver, cadence=10)
flow.add_property("sqrt(v*v + w*w)", name="Re")

# Save snapshots
if save:
    snapshots = solver.evaluator.add_file_handler(
        outpath + "snapshots", iter=rp.snapshot_iter, max_writes=5000
    )
    snapshots.add_system(solver.state)

    # Analysis tasks
    analysis = analysis_task_setup(solver, outpath, rp.analysis_iter)

    run_parameters = solver.evaluator.add_file_handler(
        outpath + "run_params", wall_dt=1e20, max_writes=1
    )
    run_parameters.add_task(a, name="a")
    run_parameters.add_task(Ny, name="Ny")
    run_parameters.add_task(Nz, name="Nz")
    run_parameters.add_task(Pr, name="Pr")
    run_parameters.add_task(Ra, name="Ra")
    run_parameters.add_task(max_dt, name="max_dt")
    run_parameters.add_task(rp.snapshot_iter, name="snapshot_iter")
    run_parameters.add_task(rp.analysis_iter, name="analysis_iter")

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
        if np.isnan(flow.max("Re")):
            raise NaNFlowError
except NaNFlowError:
    logger.error("Max Re is NaN. Triggering end of main loop.")
except KeyboardInterrupt:
    logger.info("User quit loop. Triggering end of loop.")
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
