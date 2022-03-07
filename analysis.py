"""
Author: Tom Joshi-Cale
"""
# ====================
# IMPORTS
# ====================
import numpy as np
import h5py
import argparse
import matplotlib.pyplot as plt
import pathlib
import os
import time
import imageio

from dedalus import public as de
from dedalus.tools import post

# ====================
# CLA PARSING
# ====================
parser = argparse.ArgumentParser()
parser.add_argument(
    "-i", "--input", help="Folder where the processing data is stored", required=True
)

args = parser.parse_args()

direc = os.path.normpath(args.input) + "/"

x = de.Fourier("x", 256, interval=(0, 3), dealias=3 / 2)
z = de.Chebyshev("z", 64, interval=(0, 1), dealias=3 / 2)
x = np.array(x.grid(1))
z = np.array(z.grid(1))

filenames = []
# ====================
# Plot Heatmap
# ====================
with h5py.File(direc + "snapshots/snapshots_s1/snapshots_s1_p0.h5", "r") as file:
    T = np.array(file["tasks"]["T"])
    snap_t = np.array(file["scales"]["sim_time"])
    snap_iter = np.array(file["scales"]["iteration"])

xx, zz = np.meshgrid(x, z)

maxT = np.max(T)

n_iter = len(T[:, 0:, 0])
start_time = time.time()
print("Plotting {} graphs".format(n_iter))

for i in range(0, int(n_iter)):
    fig = plt.figure(figsize=(6, 8))
    ax = fig.add_subplot(111)
    if (i % 50 == 0) and (i != 0):
        sec_per_frame = (time.time() - start_time) / i
        eta = sec_per_frame * (n_iter - i)
        print(
            "Iteration {} reached after {:.2f} seconds".format(
                i, time.time() - start_time
            )
        )
        print("Current sec_per_frame is {:.2f} seconds".format(sec_per_frame))
        print("Estimated completion in {:.2f} seconds".format(eta))

    fig.suptitle("Iteration: {}\nSim Time: {:.2f} secs".format(snap_iter[i], snap_t[i]))
    c1 = ax.contourf(
        xx, zz, np.transpose(T[i, :, :]), levels=np.linspace(0, maxT), cmap="coolwarm"
    )
    c1_bar = fig.colorbar(c1, ax=ax)
    c1_bar.set_label("T")
    ax.set_ylabel("z")
    ax.set_xlabel("x")

    plt.savefig(direc + "figure/fig_{:03d}.png".format(i))
    filenames.append(direc + "figure/fig_{:03d}.png".format(i))
    plt.close()
    plt.clf()

print("completed in {:.2f} sec".format(time.time() - start_time))

print("Creating gif...")
with imageio.get_writer(direc + "temp.gif", mode="I") as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)
