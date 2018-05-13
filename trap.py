import numpy as np
import matplotlib.pyplot as plt


def get_data(filename):
    with open(filename, "r") as f:
        raw_data = f.read()

    data = []
    for d in raw_data.split(" "):
        try:
            data.append(int(d))
        except ValueError:
            print("d = {}".format(d))

    return np.array(data)


if __name__ == "__main__":
    data = get_data("143318_ch_0_3_clr.txt")

    bg = np.sum(data[0:10]) / 10.0
    data -= bg

    x0_interpol, x1_interpol = 41, 70
    k, b = np.polyfit(np.arange(x0_interpol, x1_interpol), 
                      np.log(np.abs(data[x0_interpol:x1_interpol]) + 1),
                      1)
    print("1/k = {}, b = {}".format(1 / k, b))

    plt.plot(np.arange(len(data)), -data, "o-")

    plt.plot(np.arange(len(data)), np.log(np.abs(data) + 1), "o-")

    xp = np.arange(x0_interpol, x1_interpol)
    plt.plot(xp, k * xp + b, "-", lw = 3, c = "red")

    yp = []
    for x in xp:
        yp.append(data[x] * np.exp(-k * x))
        print("data[{}] = {}, yp = {}".format(x, data[x], yp[-1]))
    #yp = data[xp] * (1 - k * xp)
    #plt.plot(xp, yp)

    #EXP PLOT
    yp = np.exp(b) * np.exp(k * xp)
    plt.plot(xp, yp)

    #STEP-like signal
    yp = 1000 * -data[xp] / (np.exp(b) * np.exp(k * xp))
    print(yp)
    plt.plot(xp, yp)

    m_avg = 15
    avg = np.zeros(len(xp))
    for i in range(0, len(yp)):
        avg[i] = np.sum(yp[i:i + m_avg]) / m_avg
    plt.plot(xp, avg)

    plt.show()
