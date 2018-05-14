import numpy as np
import matplotlib.pyplot as plt


NUM_X = 128

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


def trap_filter(a, max_num, min_num):
    l, k = 2, 10
    
    baseline = int( np.sum(data[0:10]) / 10.0 )

    a_clear = np.array([0 if (x - baseline) > 0
                        else (x - baseline)
                        for x in a[0:min_num]])
    cTr = np.array([x for x in a_clear[0:min_num]])
    cs = np.array([x for x in cTr])


    x0_min = np.argmin(a_clear)
    print(x0_min)
    x0_interpol, x1_interpol = x0_min + 2, x0_min + 6
    ktau, b = np.polyfit(np.arange(x0_interpol, x1_interpol), 
                         np.log(-1 * a_clear[x0_interpol:x1_interpol] + 1),
                         1)
    print("-1.0 / ktau = {}".format(-1.0 / ktau))
    tau = -1.0 / ktau #3.1
    
    for i in range(1, min_num):
        cTr[i] = cTr[i - 1] + a_clear[i] - a_clear[i - 1] * np.exp(-1.0/tau)
        
    for i in range(max_num + 1, min_num):
        if i - l - k >= 0:
            cs[i] = cs[i-1] + cTr[i] - cTr[i-k] - cTr[i-l] + cTr[i-k-l]
        elif i-l >= 0:
            cs[i] = cs[i-1] + cTr[i] - cTr[i-k] - cTr[i-l]
        elif i-k >= 0:
            cs[i] = cs[i-1] + cTr[i] - cTr[i-k]
        else:
            cs[i] = cs[i-1] + cTr[i]

    res = 0.0

    return (res, a_clear, cTr, cs)


if __name__ == "__main__":
    data = get_data("143318_ch_0_3_clr.txt")

    '''
    bg = int (np.sum(data[0:10]) / 10.0)
    data -= bg

    x0_interpol, x1_interpol = 41, 70
    k, b = np.polyfit(np.arange(x0_interpol, x1_interpol), 
                      np.log(np.abs(data[x0_interpol:x1_interpol]) + 1),
                      1)
    print("1/k = {}, b = {}".format(1 / k, b))

    plt.plot(np.arange(len(data)), -data, "o-", label = "-data")

    plt.plot(np.arange(len(data)), np.log(np.abs(data) + 1), "o-", label = "log(data)")

    xp = np.arange(x0_interpol, x1_interpol)
    plt.plot(xp, k * xp + b, "-", lw = 3, c = "red", label = "exp interpol")

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

    plt.legend(loc="upper right")
    plt.show()
'''

    res, a_clear, cTr, cs = trap_filter(data, 0, 120)

    xp = np.arange(0, 120)
    plt.plot(xp, data, "o", label = "data")
    plt.plot(xp, a_clear, "-", label = "a_clear")
    plt.plot(xp, cTr, "-", label = "cTr")
    plt.plot(xp, cs, "x", label = "cs")
    
    plt.legend(loc = "upper right")
    plt.show()
