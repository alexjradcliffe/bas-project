import datetime

MINTIME = datetime.datetime(datetime.MINYEAR, 1, 1)

def read_Kp(path):
    """
    Takes a path to a .lst file containing data about Kp*10 Index downloaded
    from https://omniweb.gsfc.nasa.gov/form/dx1.html, and outputs a
    dictionary keyed on times containing the Kp data. The times are given as a
    number of hours since 00:00 on 1st January of that year.
    """
    Kp_data = {}
    i=0
    with open(path) as f:
        #NOTE - only works if all data is from same year!
        for line in f.readlines():
            line = line.split()
            time = datetime.datetime(int(line[0]), 1, 1)
            time += datetime.timedelta(days=int(line[1])-1, hours=int(line[2]))
            Kp = int(line[3])/10
            Kp_data[time] = Kp
    return Kp_data

if __name__ == "__main__":
    # Kp_data = read_Kp("sep2017/Kp/Kp_data.lst")
    Kp_data = read_Kp("day/Kp_data.lst")
    time = Kp_data.keys()
    # t0 = min(time)
    # tf = max(time)
    import numpy as np
    Kp = Kp_data.values()
    import matplotlib.pyplot as plt
    plt.plot(time, Kp)
    plt.show()
