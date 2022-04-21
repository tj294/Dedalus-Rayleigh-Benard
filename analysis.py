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

parser.add_argument(
    "-n",
    "--Nusselt",
    help="Calculate the time-averaged Nusselt Number",
    action="store_true",
)

args = parser.parse_args()

direc = os.path.normpath(args.input) + "/"

with h5py.File(direc + "run_params/run_params_s1.h5", "r") as f:
    a = int(np.array(f["tasks"]["a"]))
    Ra = float(np.array(f["tasks"]["Ra"]))
    Pr = float(np.array(f["tasks"]["Pr"]))


y = de.Fourier("y", 256, interval=(0, a), dealias=3 / 2)
z = de.Chebyshev("z", 64, interval=(0, 1), dealias=3 / 2)
y = np.array(y.grid(1))
z = np.array(z.grid(1))
# ====================
# Plot Fluxes
# ====================
if args.flux:

    avg_t_start = float(input("Start average at: "))
    avg_t_stop = float(input("End average at: "))

    with h5py.File(direc + "analysis/analysis_s1.h5", "r") as file:
        L_cond_arr = np.array(file["tasks"]["L_cond"])[:, 0]
        L_conv_arr = np.array(file["tasks"]["L_conv"])[:, 0]
        KE = np.array(file["tasks"]["KE"])[:, 0]
        snap_t = np.array(file["scales"]["sim_time"])
        Nu = np.array(file["tasks"]["Nusselt"])

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
    ASI = (np.abs(snap_t - avg_t_start)).argmin()
    if np.isnan(avg_t_stop):
        AEI = -1
    else:
        AEI = np.abs(snap_t - avg_t_stop).argmin()
    avg_t_range = snap_t[AEI] - snap_t[ASI]
    print("Averaging between {} and {}".format(snap_t[ASI], snap_t[AEI]))

    mean_L_cond = np.mean(np.array(L_cond_arr[ASI:AEI]), axis=0)
    mean_L_conv = np.mean(np.array(L_conv_arr[ASI:AEI]), axis=0)
    Nu_volume_ave = np.mean(Nu, axis=2)
    Nu_time_ave = np.mean(Nu_volume_ave[ASI:AEI], axis=0)
    print(Nu_time_ave)

    mean_L_tot = mean_L_cond + mean_L_conv
    del_L = np.max(np.abs(1.0 - mean_L_tot))
    print("max del_L = {}".format(del_L))

    fig = plt.figure(figsize=(6, 6))
    fig.suptitle("2D\nRa={:.0e}, Pr={}".format(Ra, Pr))
    KE_ax = fig.add_subplot(311)
    KE_ax.plot(snap_t, KE, "k", label="Kinetic Energy")
    KE_ax.set_xlabel(r"time [$\tau_\nu$]")
    KE_ax.set_ylabel("KE")
    KE_ax.axvspan(
        snap_t[ASI], snap_t[AEI], color="r", alpha=0.5, label="Flux averaging"
    )
    Nu_ax = KE_ax.twinx()
    Nu_ax.plot(snap_t, Nu_volume_ave, color="g", lw=0.7)
    Nu_ax.set_ylabel("Nu", color="g")
    Nu_ax.tick_params(axis="y", labelcolor="g")

    L_ax = fig.add_subplot(212)
    L_ax.plot(z, mean_L_cond, "r", linestyle="-", label=r"$L_{cond}$")
    L_ax.plot(z, mean_L_conv, "g", linestyle="-", label=r"$L_{conv}$")
    L_ax.plot(z, mean_L_tot, "k", ls="-", label=r"$L_{total}$")
    L_ax.set_xlabel("z")
    L_ax.set_ylabel("L")
    L_ax.set_title("Mean Nusselt No: {:.2f}".format(Nu_time_ave[0]))
    L_ax.legend()
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
        v = np.array(file["tasks"]["v"])
        w = np.array(file["tasks"]["w"])
        snap_t = np.array(file["scales"]["sim_time"])
        snap_iter = np.array(file["scales"]["iteration"])

    yy, zz = np.meshgrid(y, z)

    maxT = np.max(T)
    maxV = np.max(v)
    maxW = np.max(w)

    n_iter = len(T[:, 0:, 0])
    start_time = time.time()
    print("Plotting {} graphs".format(n_iter))

    try:
        for i in range(0, int(n_iter)):
            fig = plt.figure(figsize=(8, 6))
            gs = gridspec.GridSpec(ncols=2, nrows=3, figure=fig)
            T_ax = fig.add_subplot(gs[0:2, 0])
            v_ax = fig.add_subplot(gs[0, 1])
            w_ax = fig.add_subplot(gs[1, 1])
            KE_ax = fig.add_subplot(gs[2, :])
            if (i % 50 == 0) and (i != 0):
                sec_per_frame = (time.time() - start_time) / i
                eta = sec_per_frame * (n_iter - i)
                print(
                    "image {}/{} at {:.3f}ips \t | ETA in {}m {}s".format(
                        i, n_iter, sec_per_frame, int(eta // 60), int(eta % 60),
                    )
                )
            fig.suptitle(
                "2D, Ra={:.0e}, Pr={}\n".format(Ra, Pr)
                + "Iteration: {}\n".format(snap_iter[i])
                + r"Sim Time: {:.2f} $\tau_\nu$".format(snap_t[i])
            )
            c1 = v_ax.contourf(
                yy,
                zz,
                np.transpose(v[i, :, :]),
                levels=np.linspace(np.min(v), maxV),
                cmap="coolwarm",
            )
            c1_bar = fig.colorbar(c1, ax=v_ax)
            c1_bar.set_label("v", rotation=0)
            v_ax.set_ylabel("z")
            v_ax.set_xlabel("y")

            c2 = w_ax.contourf(
                yy,
                zz,
                np.transpose(w[i, :, :]),
                levels=np.linspace(np.min(w), maxW),
                cmap="coolwarm",
            )
            c2_bar = fig.colorbar(c2, ax=w_ax)
            c2_bar.set_label("w", rotation=0)
            w_ax.set_ylabel("z")
            w_ax.set_xlabel("y")

            c3 = T_ax.contourf(
                yy,
                zz,
                np.transpose(T[i, :, :]),
                levels=np.linspace(0, maxT, 11),
                cmap="coolwarm",
            )
            c3_bar = fig.colorbar(c3, ax=T_ax)
            c3_bar.set_label("T", rotation=0)
            T_ax.set_ylabel("z")
            T_ax.set_xlabel("Yy")

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
    fig.suptitle("2D\nRa={:.0e}, Pr={}".format(Ra, Pr))
    ax = fig.add_subplot(111)
    ax.plot(snap_t, KE, "k")
    ax.set_xlabel(r"time [$\tau_\kappa$]")
    ax.set_ylabel("KE")
    plt.show()
    plt.close()

if args.Nusselt:
    with h5py.File(direc + "analysis/analysis_s1.h5", "r") as f:
        Nu = np.array(f["tasks"]["Nusselt"])
        snap_t = np.array(f["scales"]["sim_time"])

    Nu_volume_ave = np.mean(Nu, axis=2)

    fig = plt.figure(figsize=(6, 4))
    fig.suptitle("2D\nRa={:.0e}, Pr={}".format(Ra, Pr))
    ax = fig.add_subplot(111)
    ax.plot(snap_t, Nu_volume_ave, "k")
    ax.set_xlabel(r"time [$\tau_\kappa$]")
    ax.set_ylabel("Nu(t)")
    plt.show()
    plt.close()


print("done.")
