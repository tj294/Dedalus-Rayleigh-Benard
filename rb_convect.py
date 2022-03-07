"""
Author: Tom Joshi-Cale
"""
import numpy as np
import logging
import os

import argparse

import dedalus.public as de
from dedalus.extras import flow_tools

logger = logging.getLogger(__name__)

# import run_params as rp
import argparse

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

args = parser.parse_args()
if args.test:
    save = False
else:
    save = True

if save:
    if "/" in args.output_folder[-1]:
        outpath = os.path.normpath(args.output_folder)
    else:
        outpath = os.path.normpath(args.output_folder + "/")
    os.makedirs(outpath, exist_ok=True)


# ====================
# Set up conditions
# ====================
a = 3
Nx, Nz = 100, 101
Pr = 0.8
Ra = 1e6

# ====================
# Create basis and domain
# ====================
xbasis = de.Fourier("x", Nx, interval(0, L), dealias=3 / 2)
