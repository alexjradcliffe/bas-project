import matplotlib.pyplot as plt
import numpy as np
import json
import datetime
from matplotlib.colors import Normalize, LogNorm
from read_Kp import read_Kp

norm = LogNorm(vmin=1e-10, vmax=1e-4)
figsize = (15,5)

MINTIME = datetime.datetime(datetime.MINYEAR, 1, 1)

def time_to_days(t):
    """
        Takes a time (t) in the form of datetime.datetime and returns the
        number of days since January 1st in datetime.MINYEAR.

            Parameters:

                t (datetime.datetime) : Any time.

            Returns:

                days (float) : The number of days by which t is after January
                1st in datetime.MINYEAR
    """
    delta = t-MINTIME
    return delta.total_seconds()/86400

def days_to_time(d):
    """
        Takes a (float) number of days and returns the datetime.datetime
        that number of days after January 1st in datetime.MINYEAR.

            Parameters:

                days (float) : Any positive number of days.

            Returns:

                t (datetime.datetime) : The time d days after January 1st in
                datetime.MINYEAR
    """
    return MINTIME + datetime.timedelta(days=d)

if __name__ == "__main__":
    dir_path = "example"
    with open(dir_path + '/models_input.json', "r") as inputJSON:
        models_input = json.load(inputJSON)
    L_range = models_input["L_range"]
    t_range = models_input["t_range"]
    print(t_range)
    VAP_PSD = np.array(models_input["PSD"])
    VAP_points = np.array(models_input["VAP_points"])
    with open(dir_path + '/models_output.json', "r") as outputJSON:
        models_output = json.load(outputJSON)
    diffusion_PSD = np.array(models_output["diffusion_output"])
    kalman_PSD = np.array(models_output["kalman_output"])
    innovation = np.array(models_output["innovation"])

    def pcolor_data_from_PSD(PSD):
        nT, nL = PSD.shape
        times = np.vectorize(days_to_time)(np.linspace(t_range[0], t_range[1], nT))
        Li = np.linspace(L_range[0], L_range[1], nL)
        X = np.array([times for i in range(nL)])
        Y = np.transpose([Li for i in range(nT)])
        Z = np.transpose(PSD)
        assert X.shape == Y.shape == Z.shape
        return X, Y, Z

    Kp_data = read_Kp(dir_path + "/Kp_data.lst")
    Kp_times = list(sorted(Kp_data.keys()))
    Kp_times = [t for t in Kp_times if days_to_time(t_range[0]) <= t <=
                days_to_time(t_range[1])]
    Kp_values = [Kp_data[t] for t in Kp_times]
    assert(Kp_times != [])
    plt.figure(figsize=figsize)
    plt.title("Kp data")
    plt.plot(Kp_times, Kp_values)
    plt.xlabel('Time')
    plt.ylabel('Kp')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(dir_path + "/Kp.png")
    plt.show()

    X, Y, Z = pcolor_data_from_PSD(VAP_PSD)
    plt.figure(figsize=figsize)
    c = plt.pcolor(X, Y, Z, norm=norm, cmap=plt.cm.rainbow)
    cbar = plt.colorbar(c)
    cbar.set_label('Density $(c/(cm MeV))^3/sr$', rotation=270)
    plt.title("Interpolated Van Allen Probe PSD")
    plt.xlabel('Time')
    plt.ylabel('L $(R_E)$')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(dir_path + "/VAP_interpolated.png")
    plt.show()

    X, Y, Z = pcolor_data_from_PSD(VAP_points)
    plt.figure(figsize=figsize)
    c = plt.pcolor(X, Y, Z, norm=norm, cmap=plt.cm.rainbow)
    cbar = plt.colorbar(c)
    cbar.set_label('Density $(c/(cm MeV))^3/sr$', rotation=270)
    plt.title("Van Allen Probe Point Densities")
    plt.xlabel('Time')
    plt.ylabel('L $(R_E)$')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(dir_path + "/VAP_points.png")
    plt.show()

    X, Y, Z = pcolor_data_from_PSD(diffusion_PSD)
    plt.figure(figsize=figsize)
    plt.title("Diffusion Model PSD")
    c = plt.pcolor(X, Y, Z, norm=norm, cmap=plt.cm.rainbow)
    cbar = plt.colorbar(c)
    cbar.set_label('Density $(c/(cm MeV))^3/sr$', rotation=270)
    plt.xlabel('Time')
    plt.ylabel('L $(R_E)$')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(dir_path + "/Diffusion_model.png")
    plt.show()

    X, Y, Z = pcolor_data_from_PSD(kalman_PSD)
    plt.figure(figsize=figsize)
    c=plt.pcolor(X, Y, Z, norm=norm, cmap=plt.cm.rainbow)
    cbar = plt.colorbar(c)
    plt.title("Kalman Model PSD")
    cbar.set_label('Density $(c/(cm MeV))^3/sr$', rotation=270)
    plt.xlabel('Time')
    plt.ylabel('L $(R_E)$')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(dir_path + "/Kalman_model.png")
    plt.show()

    Li = np.linspace(L_range[0], L_range[1], innovation.shape[1])
    plt.figure(figsize=figsize)
    plt.plot(Li, np.nanmean(innovation, axis=0))
    plt.title("EnKF Innovation averaged through time")
    plt.xlabel('L $(R_E)$')
    plt.ylabel('Innovation')
    plt.savefig(dir_path + '/EnKF_innovation.png')
    plt.show()
