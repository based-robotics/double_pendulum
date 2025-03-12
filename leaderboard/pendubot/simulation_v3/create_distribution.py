import argparse
import importlib
import os
from importlib import reload

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots
from double_pendulum.analysis.leaderboard import get_number_of_swingups, get_uptime
from double_pendulum.controller.global_policy_testing_controller import (
    GlobalPolicyTestingControllerV2,
)
from double_pendulum.model.plant import DoublePendulumPlant
from double_pendulum.simulation.simulation import Simulator
from double_pendulum.utils.csv_trajectory import load_trajectory_full, save_trajectory
from double_pendulum.utils.plotting import plot_timeseries
from sim_parameters import (
    dt,
    eps,
    goal,
    integrator,
    knockdown_after,
    knockdown_length,
    method,
    mpar,
    mpar_nolim,
    t0,
    t_final,
    x0,
)

plt.style.use(["science", "ieee", "no-latex"])


def simulate_controller(controller, save_dir, traj_file_name=""):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    plant = DoublePendulumPlant(model_pars=mpar_nolim)
    sim = Simulator(plant=plant)

    T, X, U = sim.simulate(
        t0=t0,
        x0=x0,
        tf=t_final,
        dt=dt,
        controller=controller,
        integrator=integrator,
    )
    traj_file_name = os.path.join(save_dir, f"{traj_file_name}.csv")
    save_trajectory(traj_file_name, T=T, X_meas=X, U_con=U)


def leaderboard_scores(
    data_paths,
    mpar,
    weights,
    normalize,
    link_base="",
    simulation=True,
    score_version="v1",
    t_final=10.0,
):
    leaderboard_data = []

    if "uptime" not in weights.keys():
        weights["uptime"] = 0.0
    if "uptime" not in normalize.keys():
        normalize["uptime"] = 1.0

    results_dict = {}
    for key in data_paths:
        d = data_paths[key]
        if type(d["csv_path"]) == str:
            csv_paths = [d["csv_path"]]
        else:
            csv_paths = d["csv_path"]

        n_swingups = []
        uptimes = []
        scores = []

        for path in sorted(csv_paths):
            data_dict = load_trajectory_full(path)
            T = data_dict["T"]
            X = data_dict["X_meas"]
            U = data_dict["U_con"]

            n_swingups.append(get_number_of_swingups(T, X, mpar=mpar, method="height", height=0.9))
            uptimes.append(get_uptime(T, X, mpar=mpar, method="height", height=0.9))

            score = weights["uptime"] * uptimes[-1] / normalize["uptime"]

            scores.append(score)

            header = ""
            results = []

            if weights["uptime"] != 0.0:  # intentionally checking for uptime
                results.append([n_swingups[-1]])
                header += "#swingups,"
            if weights["uptime"] != 0.0:
                results.append([uptimes[-1]])
                header += "Uptime [s],"
            results.append([score])
            header += "RealAI Score"
            results = np.asarray(results).T

        # best = np.argmax(scores)
        # uptime = uptimes[best]
        # n_swingup = n_swingups[best]
        # score = np.mean(scores)
        # best_score = np.max(scores)

        if link_base != "":
            if "simple_name" in d.keys():
                name_with_link = f"[{d['simple_name']}]({link_base}{d['name']}/README.md)"
            else:
                name_with_link = f"[{d['name']}]({link_base}{d['name']}/README.md)"
        else:
            if "simple_name" in d.keys():
                name_with_link = d["simple_name"]
            else:
                name_with_link = d["name"]

        append_data = [name_with_link, d["short_description"]]

        if weights["uptime"] != 0.0:  # intentionally checking for uptime
            append_data.append(n_swingups)
        if weights["uptime"] != 0.0:
            append_data.append(np.round(uptimes, 3))

        if simulation:
            append_data.append(np.round(score, 3))
            append_data.append(d["username"])

            if link_base != "":
                controller_link = link_base + d["name"]

                data_link = "[data](" + controller_link + "/sim_swingup.csv)"
                plot_link = "[plot](" + controller_link + "/timeseries.png)"
                video_link = "[video](" + controller_link + "/sim_video.gif)"
                append_data.append(data_link + " " + plot_link + " " + video_link)
        else:
            append_data.append(str(round(best_score, 3)))
            append_data.append(str(round(score, 3)))
            append_data.append(d["username"])
            if link_base != "":
                controller_link = link_base + d["name"]
                data_link = "[data](" + controller_link + "/experiment" + str(best + 1).zfill(2) + "/trajectory.csv)"
                plot_link = "[plot](" + controller_link + "/experiment" + str(best + 1).zfill(2) + "/timeseries.png)"
                video_link = "[video](" + controller_link + "/experiment" + str(best + 1).zfill(2) + "/video.gif)"
                append_data.append(data_link + " " + plot_link + " " + video_link)

        leaderboard_data.append(append_data)

    header = "Controller,"
    header += "Short Controller Description,"
    if weights["uptime"] != 0.0:  # intentionally checking for uptime
        header += "#swingups,"
    if weights["uptime"] != 0.0:
        header += "Uptime [s],"

    if simulation:
        header += "RealAI Score,"
        header += "Username"
    else:
        header += "Best RealAI Score,"
        header += "Average RealAI Score,"
        header += "Username"

    if link_base != "":
        header += ",Data"

    return header, leaderboard_data


parser = argparse.ArgumentParser()
parser.add_argument(
    "--data-dir",
    dest="data_dir",
    help="Directory for saving data. Existing data will be kept.",
    default="dist_data",
    required=False,
)
parser.add_argument(
    "--num-iters",
    dest="num_iters",
    help="Directory for saving data. Existing data will be kept.",
    default=2,
    required=False,
)
parser.add_argument(
    "--gen-data",
    dest="gen_data",
    help="Whether to generate data.",
    default=True,
    required=False,
)

gen_data = parser.parse_args().gen_data
data_dir = parser.parse_args().data_dir
num_iters = int(parser.parse_args().num_iters)
stats = {}

if gen_data:
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    for file in os.listdir("."):
        if file[:4] == "con_":
            print(f"Simulating new controller {file}")
            controller_arg = file[:-3]
            controller_name = controller_arg[4:]

            save_dir = os.path.join(data_dir, f"{controller_name}")
            if os.path.exists(save_dir):
                print(f"Skipping {controller_name}")
                continue
            else:
                os.makedirs(save_dir)

            imp = importlib.import_module(controller_arg)
            for batch_idx in range(num_iters):
                reload(imp)

                swingup_controller = imp.controller
                global_policy_testing_controller = GlobalPolicyTestingControllerV2(
                    swingup_controller,
                    goal=goal,
                    # knockdown_after=knockdown_after,
                    # knockdown_length=knockdown_length,
                    n_disturbances=15,
                    t_max=t_final,
                    reset_length=0.2,
                    # method=method,
                    # eps=eps,
                    mpar=mpar_nolim,
                )
                simulate_controller(
                    global_policy_testing_controller,
                    save_dir=save_dir,
                    traj_file_name=f"{controller_name}_{batch_idx + 1}",
                )
                print(f"    {batch_idx + 1}/{num_iters}")

src_dir = "."
leaderboard_configs = {}
# Iterate over controller modules, expecting filenames starting with "con_"
for f in os.listdir(src_dir):
    if f.startswith("con_") and f.endswith(".py"):
        mod = importlib.import_module(f[:-3])
        # Expect each module to provide a leaderboard_config dict with keys "name" and "csv_path".
        if hasattr(mod, "leaderboard_config"):
            # Build the full path to the CSV file within the data directory. (The csv_path key can be a relative path, e.g. "controller_1/controller_1_1.csv")
            controller = f[4:-3]
            pathes = []
            for f_con in os.listdir(os.path.join(data_dir, controller)):
                pathes.append(os.path.join(data_dir, controller, f_con))
            conf = mod.leaderboard_config
            conf["csv_path"] = pathes
            leaderboard_configs[mod.leaderboard_config["name"]] = conf

# for controller_name, leaderboard_data in leaderboard_configs.items():
header, leaderboard = leaderboard_scores(
    data_paths=leaderboard_configs,
    mpar=mpar,
    weights={"uptime": 1.0},
    normalize={"uptime": t_final},
    score_version="v3",
)
# Sort the leaderboard entries by the median uptime (ascending order)
leaderboard_sorted = sorted(leaderboard, key=lambda r: np.median(r[3]), reverse=True)

# Extract sorted controller names
controllers = [entry[0] for entry in leaderboard_sorted]

# Compute summary metrics for swingups and uptime per controller
swingups_mean = [np.mean(entry[2]) for entry in leaderboard_sorted]
swingups_std = [np.std(entry[2]) for entry in leaderboard_sorted]

uptime_mean = [np.mean(entry[3]) for entry in leaderboard_sorted]
uptime_std = [np.std(entry[3]) for entry in leaderboard_sorted]

x_pos = np.arange(len(controllers))

# Create two subplots, one for swingups and one for uptime
fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True)

# Plot Swingups (top subplot)
ax2.bar(x_pos, swingups_mean, yerr=swingups_std, capsize=5, color="skyblue")
ax2.set_ylabel("Swingups")
ax2.grid(True, axis="y", linestyle="--", alpha=0.7)

# Plot Uptime (bottom subplot)
ax1.bar(x_pos, uptime_mean, yerr=uptime_std, capsize=5, color="lightgreen")
ax1.set_ylabel("Uptime [s]")
ax1.grid(True, axis="y", linestyle="--", alpha=0.7)

# Set x-axis tick labels using sorted controller names
plt.xticks(x_pos, controllers, rotation=45, ha="right")
plt.tight_layout()

# Save the plot with high DPI
plt.savefig("sorted_controllers_comparison.png", dpi=300)
plt.show()


# np.savetxt(
#     "batch_results.csv",
#     leaderboard_sorted,
#     header=header,
#     delimiter=",",
#     fmt="%s",
#     comments="",
# )
