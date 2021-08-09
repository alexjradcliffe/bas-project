def read_Kp(path):
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
            Kp = int(line[3])
            Kp_data[hour] = Kp
            tf = hour
    assert(tf - t0+1 == len(Kp_data))
    print(t0, tf)
    return (t0, tf, Kp_data)

if __name__ == "__main__":
    t0, tf, Kp_data = read_Kp("sep2017/Kp/Kp_data.lst")
    import numpy as np
    time = np.array(list(Kp_data.keys()))
    Kp = np.array(list(Kp_data.values()))
    import matplotlib.pyplot as plt
    plt.plot(time, Kp)
    plt.show()
