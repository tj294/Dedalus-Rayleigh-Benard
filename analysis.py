"""
Analysis code for plotting vertical flux transport and/or a gif of temperature,
velocity and KE from the merged output of a Dedalus Rayleigh-Bérnard code.
Author: Tom Joshi-Cale
"""
# ====================
# IMPORTS
# ====================
import numpy as np
import h5py
import argparse
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pathlib
import os
import shutil
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
parser.add_argument(
    "-t", "--heatmap", help="Plot a gif of the temperature heatmap", action="store_true"
)
parser.add_argument(
    "-f", "--flux", help="Plot the average flux contributions", action="store_true"
)
parser.add_argument(
    "-k", "--KE", help="Plot the kinetic energy only", action="store_true"
)

args = parser.parse_args()

direc = os.path.normpath(args.input) + "/"


x = de.Fourier("x", 256, interval=(0, 3), dealias=3 / 2)
z = de.Chebyshev("z", 64, interval=(0, 1), dealias=3 / 2)
x = np.array(x.grid(1))
z = np.array(z.grid(1))
# ====================
# Plot Fluxes
# ====================
avg_t_start = 0.6
avg_t_stop = 1.3

if args.flux:
    with h5py.File(direc + "analysis/analysis_s1.h5", "r") as file:
        L_cond_arr = np.array(file["tasks"]["L_cond"])[:, 0]
        L_conv_arr = np.array(file["tasks"]["L_conv"])[:, 0]
        snap_t = np.array(file["scales"]["sim_time"])

    print(L_cond_arr)

    if (
        (avg_t_start <= snap_t[0])
        or (avg_t_start >= snap_t[-1])
        or (avg_t_stop <= snap_t[0])
        or (avg_t_stop >= snap_t[-1])
    ):
        print(
            "Average time period out of simulation range: {} -> {}".format(
                snap_t[0], snap_t[-1]
            )
        )
        pass
    ASI = (np.abs(snap_t - avg_t_stop)).argmin()
    if np.isnan(avg_t_stop):
        AEI = -1
    else:
        AEI = np.abs(snap_t - avg_t_stop).argmin()
    avg_t_range = snap_t[AEI] - snap_t[ASI]

    mean_L_cond = np.mean(np.array(L_cond_arr), axis=0)
    mean_L_conv = np.mean(np.array(L_conv_arr), axis=0)

    mean_L_tot = mean_L_cond + mean_L_conv
    del_L = np.max(np.abs(1.0 - mean_L_tot))
    print("max del_L = {}".format(del_L))

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    ax.plot(z, mean_L_cond, "r", linestyle="-", label=r"$L_{cond}$")
    ax.plot(z, mean_L_conv, "g", linestyle="-", label=r"$L_{conv}$")
    ax.plot(z, mean_L_tot, "k", ls="-", label=r"$L_{total}$")
    ax.set_xlabel("z")
    ax.set_ylabel("L")
    ax.legend()
    plt.savefig(direc + "fluxes.png")
    plt.show()
    plt.close()

# ====================
# Plot heatmap
# ====================

if args.heatmap:
    filenames = []

    os.makedirs(direc + "figure", exist_ok=True)

    with h5py.File(direc + "analysis/analysis_s1.h5", "r") as file:
        KE = np.array(file["tasks"]["KE"])[:, 0]
    with h5py.File(direc + "snapshots/snapshots_s1.h5", "r") as file:
        T = np.array(file["tasks"]["T"])
        u = np.array(file["tasks"]["u"])
        w = np.array(file["tasks"]["w"])
        snap_t = np.array(file["scales"]["sim_time"])
        snap_iter = np.array(file["scales"]["iteration"])

    xx, zz = np.meshgrid(x, z)

    maxT = np.max(T)
    maxU = np.max(u)
    maxW = np.max(w)

    n_iter = len(T[:, 0:, 0])
    start_time = time.time()
    print("Plotting {} graphs".format(n_iter))

    try:
        for i in range(0, int(n_iter)):
            fig = plt.figure(figsize=(8, 6))
            gs = gridspec.GridSpec(ncols=2, nrows=3, figure=fig)
            T_ax = fig.add_subplot(gs[0:2, 0])
            u_ax = fig.add_subplot(gs[0, 1])
            v_ax = fig.add_subplot(gs[1, 1])
            KE_ax = fig.add_subplot(gs[2, :])
            if (i % 50 == 0) and (i != 0):
                sec_per_frame = (time.time() - start_time) / i
                eta = sec_per_frame * (n_iter - i)
                print(
                    "image {}/{} at {:.3f}ips \t| ETA in {}m {}s".format(
                        i, n_iter, sec_per_frame, int(eta // 60), int(eta % 60),
                    ),
                    end="\r",
                )

            fig.suptitle(
                "Iteration: {}\n".format(snap_iter[i])
                + r"Sim Time: {:.2f} $\tau_\kappa$".format(snap_t[i])
            )
            c1 = u_ax.contourf(
                xx,
                zz,
                np.transpose(u[i, :, :]),
                levels=np.linspace(np.min(u), maxU),
                cmap="coolwarm",
            )
            c1_bar = fig.colorbar(c1, ax=u_ax)
            c1_bar.set_label("u", rotation=0)
            u_ax.set_ylabel("z")
            u_ax.set_xlabel("x")

            c2 = v_ax.contourf(
                xx,
                zz,
                np.transpose(w[i, :, :]),
                levels=np.linspace(np.min(w), maxW),
                cmap="coolwarm",
            )
            c2_bar = fig.colorbar(c2, ax=v_ax)
            c2_bar.set_label("w", rotation=0)
            v_ax.set_ylabel("z")
            v_ax.set_xlabel("x")

            c3 = T_ax.contourf(
                xx,
                zz,
                np.transpose(T[i, :, :]),
                levels=np.linspace(0, maxT),
                cmap="coolwarm",
            )
            c3_bar = fig.colorbar(c3, ax=T_ax)
            c3_bar.set_label("T", rotation=0)
            T_ax.set_ylabel("z")
            T_ax.set_xlabel("x")

            KE_ax.plot(snap_t[:i], KE[:i], "k")
            KE_ax.set_xlabel(r"time [$\tau_\kappa$]")
            KE_ax.set_ylabel("KE")
            KE_ax.set_ylim([0, 1.1 * np.max(KE)])
            KE_ax.set_xlim([0, np.max(snap_t)])

            plt.tight_layout()
            plt.savefig(direc + "figure/fig_{:03d}.png".format(i))
            filenames.append(direc + "figure/fig_{:03d}.png".format(i))
            plt.close()
            plt.clf()
    except KeyboardInterrupt:
        print("ending loop")

    print("completed in {:.2f} sec".format(time.time() - start_time))

    print("Creating gif...")
    with imageio.get_writer(direc + "info.gif", mode="I") as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)

    print("Removing raw image files...")
    shutil.rmtree(direc + "figure")

if args.KE:
    with h5py.File(direc + "analysis/analysis_s1.h5", "r") as f:
        KE = np.array(f["tasks"]["KE"])[:, 0]
        snap_t = np.array(f["scales"]["sim_time"])
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    ax.plot(snap_t, KE, "k")
    ax.set_xlabel(r"time [$\tau_\kappa$]")
    ax.set_label("KE")
    plt.show()
    plt.close()

print("done.")
