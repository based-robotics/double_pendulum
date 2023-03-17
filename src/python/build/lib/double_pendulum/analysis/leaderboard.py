import numpy as np

from double_pendulum.utils.csv_trajectory import load_trajectory_full


def leaderboard_scores(
    data_paths,
    save_to,
    weights={
        "swingup_time": 0.2,
        "max_tau": 0.1,
        "energy": 0.0,
        "integ_tau": 0.1,
        "tau_cost": 0.0,
        "tau_smoothness": 0.6,
    },
    normalize={
        "swingup_time": 10.0,
        "max_tau": 1.0,
        "energy": 1.0,
        "integ_tau": 10.0,
        "tau_cost": 10.0,
        "tau_smoothness": 1.0,
    },
):
    """leaderboard_scores.
    Compute leaderboard scores from data_dictionaries which will be loaded from
    data_paths.  Data can be either from simulation or experiments (but for
    comparability it should only be one).

    Parameters
    ----------
    data_paths : dict
        contains the names and paths to the trajectory data in the form:
        {controller1_name: {"csv_path": data_path1, "name": controller1_name, "username": username1},
         controller2_name: {"csv_path": data_path2, "name": controller2_name, "username": username2},
         ...}
    save_to : string
        path where the result will be saved as a csv file
    weights : dict
        dictionary containing the weights for the different criteria in the
        form:
        {"swingup_time": weight1,
         "max_tau": weight2,
         "energy": weight3,
         "integ_tau": weight4,
         "tau_cost": weight5,
         "tau_smoothness": weight6}
         The weights should sum up to 1 for the final score to be in the range
         [0, 1].
    normalize : dict
        dictionary containing normalization constants for the different
        criteria in the form:
        {"swingup_time": norm1,
         "max_tau": norm2,
         "energy": norm3,
         "integ_tau": norm4,
         "tau_cost": norm5,
         "tau_smoothness": norm6}
         The normalization constants should be the maximum values that can be
         achieved by the criteria so that after dividing by the norm the result
         is in the range [0, 1].
    simulation : bool
        whether to load the simulaition trajectory data
    """

    leaderboard_data = []

    for key in data_paths:
        d = data_paths[key]
        data_dict = load_trajectory_full(d["csv_path"])
        T = data_dict["T"]
        X = data_dict["X_meas"]
        U = data_dict["U_con"]

        swingup_time = get_swingup_time(T, X)
        max_tau = get_max_tau(U)
        energy = get_energy(X, U)
        integ_tau = get_integrated_torque(T, U)
        tau_cost = get_torque_cost(T, U)
        tau_smoothness = get_tau_smoothness(U)

        score = (
            weights["swingup_time"] * swingup_time / normalize["swingup_time"]
            + weights["max_tau"] * max_tau / normalize["max_tau"]
            + weights["energy"] * energy / normalize["energy"]
            + weights["integ_tau"] * integ_tau / normalize["integ_tau"]
            + weights["tau_cost"] * tau_cost / normalize["tau_cost"]
            + weights["tau_smoothness"] * tau_smoothness / normalize["tau_smoothness"]
        )

        score = 1 - score

        leaderboard_data.append(
            [
                d["name"],
                str(swingup_time),
                str(energy),
                str(max_tau),
                str(integ_tau),
                str(tau_cost),
                str(tau_smoothness),
                str(score),
                d["username"],
            ]
        )

    np.savetxt(
        save_to,
        leaderboard_data,
        header="Controller,Swingup Time,Energy,Max. Torque,Integrated Torque,Torque Cost,Torque Smoothness,Real AI Score,Username",
        delimiter=",",
        fmt="%s",
        comments="",
    )


def get_swingup_time(T, X, eps=[1e-2, 1e-2, 1e-2, 1e-2], has_to_stay=True):
    """get_swingup_time.
    get the swingup time from a data_dict.

    Parameters
    ----------
    T : array-like
        time points, unit=[s]
        shape=(N,)
    X : array-like
        shape=(N, 4)
        states, units=[rad, rad, rad/s, rad/s]
        order=[angle1, angle2, velocity1, velocity2]
    U : array-like
        shape=(N, 2)
        actuations/motor torques
        order=[u1, u2],
        units=[Nm]
    eps : list
        list with len(eps) = 4. The thresholds for the swingup to be
        successfull ([position, velocity])
        default = [1e-2, 1e-2, 1e-2, 1e-2]
    has_to_stay : bool
        whether the pendulum has to stay upright until the end of the trajectory
        default=True

    Returns
    -------
    float
        swingup time
    """
    print("X", np.shape(X))
    print("X.T[0]", np.shape(X.T[0]))
    goal = np.array([np.pi, 0.0, 0.0, 0.0])

    dist_x0 = np.abs(np.mod(X.T[0], 2 * np.pi) - goal[0])
    ddist_x0 = np.where(dist_x0 < eps[0], 0.0, dist_x0)
    n_x0 = np.argwhere(ddist_x0 == 0.0)

    dist_x1 = np.abs(np.mod(X.T[1], 2 * np.pi) - goal[1])
    ddist_x1 = np.where(dist_x1 < eps[1], 0.0, dist_x1)
    n_x1 = np.argwhere(ddist_x1 == 0.0)

    dist_x2 = np.abs(X.T[2] - goal[2])
    ddist_x2 = np.where(dist_x2 < eps[2], 0.0, dist_x2)
    n_x2 = np.argwhere(ddist_x2 == 0.0)

    dist_x3 = np.abs(X.T[3] - goal[3])
    ddist_x3 = np.where(dist_x3 < eps[3], 0.0, dist_x3)
    n_x3 = np.argwhere(ddist_x3 == 0.0)

    n = np.intersect1d(n_x0, n_x1)
    n = np.intersect1d(n, n_x2)
    n = np.intersect1d(n, n_x3)

    time_index = len(T) - 1
    if has_to_stay:
        if len(n) > 0:
            for i in range(len(n) - 2, 0, -1):
                if n[i] + 1 == n[i + 1]:
                    time_index = n[i]
                else:
                    break
    else:
        if len(n) > 0:
            time_index = n[0]
    time = T[time_index]

    return time


def get_max_tau(U):
    """get_max_tau.

    Get the maximum torque used in the trajectory.

    Parameters
    ----------
    U : array-like
        shape=(N, 2)
        actuations/motor torques
        order=[u1, u2],
        units=[Nm]

    Returns
    -------
    float
        maximum torque
    """
    tau = np.max(np.abs(U))
    return tau


def get_energy(X, U):
    """get_energy.

    Get the mechanical energy used during the swingup.

    Parameters
    ----------
    X : array-like
        shape=(N, 4)
        states, units=[rad, rad, rad/s, rad/s]
        order=[angle1, angle2, velocity1, velocity2]
    U : array-like
        shape=(N, 2)
        actuations/motor torques
        order=[u1, u2],
        units=[Nm]

    Returns
    -------
    float
        energy
    """

    delta_pos0 = np.diff(X.T[0])
    tau0 = U.T[0][:-1]
    energy0 = np.sum(np.abs(delta_pos0 * tau0))

    delta_pos1 = np.diff(X.T[1])
    tau1 = U.T[1][:-1]
    energy1 = np.sum(np.abs(delta_pos1 * tau1))

    energy = energy0 + energy1

    return energy


def get_integrated_torque(T, U):
    """get_integrated_torque.

    Get the (discrete) time integral over the torque.

    Parameters
    ----------
    T : array-like
        time points, unit=[s]
        shape=(N,)
    U : array-like
        shape=(N, 2)
        actuations/motor torques
        order=[u1, u2],
        units=[Nm]

    Returns
    -------
    float
        integrated torque
    """
    delta_t = np.diff(T)
    tau0 = np.abs(U.T[0][:-1])
    int_tau0 = np.sum(tau0 * delta_t)

    tau1 = np.abs(U.T[1][:-1])
    int_tau1 = np.sum(tau1 * delta_t)

    int_tau = int_tau0 + int_tau1

    return int_tau


def get_torque_cost(T, U, R=np.diag([1.0, 1.0])):
    """get_torque_cost.

    Get the running cost torque with cost parameter R.
    The cost is normalized with the timestep.

    Parameters
    ----------
    U : array-like
        shape=(N, 2)
        actuations/motor torques
        order=[u1, u2],
        units=[Nm]
    R : numpy array
        running cost weight matrix (2x2)

    Returns
    -------
    float
        torque cost
    """
    delta_t = np.diff(T)
    cost = np.einsum("ij, i, jk, ik", U[:-1], delta_t, R, U[:-1])
    return cost


def get_tau_smoothness(U):
    """get_tau_smoothness.

    Get the standard deviation of the changes in the torque signal.

    Parameters
    ----------
    U : array-like
        shape=(N, 2)
        actuations/motor torques
        order=[u1, u2],
        units=[Nm]

    Returns
    -------
    float
        torque smoothness (std of changes)
    """
    u_diff0 = np.diff(U.T[0])
    std0 = np.std(u_diff0)

    u_diff1 = np.diff(U.T[1])
    std1 = np.std(u_diff1)

    std = std0 + std1

    return std
