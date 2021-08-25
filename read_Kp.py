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
            hour = int(line[2])+24*(int(line[1])-1)
            if i == 0:
                t0 = hour
            i += 1
            Kp = int(line[3])/10
            Kp_data[hour] = Kp
            tf = hour
    assert(tf - t0+1 == len(Kp_data))
    return Kp_data

if __name__ == "__main__":
    # Kp_data = read_Kp("sep2017/Kp/Kp_data.lst")
    Kp_data = read_Kp("week/Kp_data.lst")
    t0 = min(Kp_data.keys())
    tf = min(Kp_data.keys())
    import numpy as np
    time = [key/24 + 1 for key in Kp_data.keys()]
    Kp = Kp_data.values()
    import matplotlib.pyplot as plt
    plt.plot(time, Kp)
    plt.show()
