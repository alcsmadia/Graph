import numpy as np # 数学ライブラリ
import matplotlib.pyplot as plt # プロット
from scipy.interpolate import griddata

table = np.genfromtxt("Table.csv", delimiter=",", dtype='float', skip_header=1, \
                         names=["dbdt", "B", "Bm", "Hm", "Hh", "Hf", "Hhf"])
real  = np.genfromtxt("real.csv", delimiter=",", dtype='float', skip_header=1, \
                         names=["t", "dbdt", "B", "H", "Hm", "Hhf"])
half  = len(table/40)
"""
def saigen(Z):
     x, y = np.meshgrid(real['dbdt'], real['B'], sparse = True)
     z    = np.diag(griddata((table['dbdt'], table['B']), Z, (x, y), method='linear'))
     return (z)

"""
def saigen(Z):
    x0 = y0 = z0 = np.empty(0)
    for ii in range(0, len(real)):
        data_dbdt = np.array([5, 10, 20, 30, 40, 50, 60, 80, 100, 125, 150, \
                              175, 200, 300, 400, 500, 600, 700, 800, 900], dtype=object)
        data_dbdt = data_dbdt / 1000
        x = real['dbdt'][ii]  # float(input("dbdt:"))
        dbdt_close = (np.abs(data_dbdt - x)).argsort()[:2]  # xに近い順にdbdtを2点取りだす
        dbdt_close = np.sort(dbdt_close)  # 取り出したdbdtを昇順にする

        y = real['B'][ii]  # float(input("B   :"))
        # yに近い順にBを2*2点取り出す
        B_close1 = (np.abs(table['B'][dbdt_close[0] * half: (dbdt_close[0] + 1) * half] - y)).argsort()[:2] + dbdt_close[0] * half
        B_close2 = (np.abs(table['B'][dbdt_close[1] * half: (dbdt_close[1] + 1) * half] - y)).argsort()[:2] + dbdt_close[1] * half
        B_close1, B_close2 = np.sort(B_close1), np.sort(B_close2)
        
        x1, y1, z1 = table['dbdt'][B_close1[0]], table['B'][B_close1[0]], Z[B_close1[0]] # 再現するdBdt, Bに近い点を4つ取り出す
        x2, y2, z2 = table['dbdt'][B_close1[1]], table['B'][B_close1[1]], Z[B_close1[1]]
        x3, y3, z3 = table['dbdt'][B_close2[0]], table['B'][B_close2[0]], Z[B_close2[0]]
        x4, y4, z4 = table['dbdt'][B_close2[1]], table['B'][B_close2[1]], Z[B_close2[1]]

        if y > (y4 - y1) / (x4 - x1) * (x - x1) + y1: # Bが斜め線の上にあるか下にあるか
            z = ((x1 - x) * ((y2 - y1) * (z4 - z1) - (y4 - y1) * (z2 - z1)) + (y1 - y) * (
                        (z2 - z1) * (x4 - x1) - (z4 - z1) * (x2 - x1))) / (
                            (x2 - x1) * (y4 - y1) - (x4 - x1) * (y2 - y1)) + z1 # 三点を通る平面の式
        else:
            z = ((x1 - x) * ((y2 - y1) * (z3 - z1) - (y3 - y1) * (z2 - z1)) + (y1 - y) * (
                        (z2 - z1) * (x3 - x1) - (z3 - z1) * (x2 - x1))) / (
                            (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)) + z1
        z0 = np.append(z0, z)
    return (z0)

t = np.concatenate((real['t'], -1 * real['t'][::-1]))
def exchange2(B): #
    return(np.concatenate((B, -1 * B)))

# 実測
dbdt, B = exchange2(real['dbdt']), exchange2(real['B'])
Hm, Hhf = exchange2(real['Hm'])  , exchange2(real['Hhf'])
H       = exchange2(real['H'])

# 再現
Hm_sim  = exchange2(saigen(table['Hm']))
Hhf_sim = exchange2(saigen(table['Hhf']))
H_sim   = Hm_sim + Hhf_sim

# --------- 描写 -----------
fig = plt.figure(figsize=(10,4)) # グラフを表示する(5,4), (24,4)
fig.subplots_adjust(wspace=0.3)

# iの成分
ax4 = fig.add_subplot(1, 2, 2)
ax4.plot(t, exchange2(real['Hm']), marker="None", label="$H_m$", color='skyblue', linewidth = 0.5)
ax4.plot(t, Hm_sim, marker="None", label="$H_m(Sim.)$", color='b', linewidth = 0.5)
ax4.plot(t, exchange2(real['Hhf']), marker="None", label="$H_h+H_f$", color='pink', linewidth = 0.5)
ax4.plot(t, Hhf_sim, marker="None", label="$H_h+H_f(Sim.)$", color='purple', linewidth = 0.5)
ax5 = ax4.twinx()
ax5.plot(t, dbdt, marker="None", label="$dbdt$", color='grey', linewidth = 0.5)
"""
ax5.axhline(y=5/1000, color='r', linestyle='--', linewidth = 0.5, alpha=0.4)
ax5.axhline(y=10/1000, color='r', linestyle='--', linewidth = 0.5, alpha=0.4)
ax5.axhline(y=20/1000, color='r', linestyle='--', linewidth = 0.5, alpha=0.4)
ax5.axhline(y=30/1000, color='r', linestyle='--', linewidth = 0.5, alpha=0.4)
ax5.axhline(y=40/1000, color='r', linestyle='--', linewidth = 0.5, alpha=0.4)
ax5.axhline(y=50/1000, color='r', linestyle='--', linewidth = 0.5, alpha=0.4)
ax5.axhline(y=60/1000, color='r', linestyle='--', linewidth = 0.5, alpha=0.4)
ax5.axhline(y=80/1000, color='r', linestyle='--', linewidth = 0.5, alpha=0.4)
ax5.axhline(y=100/1000, color='r', linestyle='--', linewidth = 0.5, alpha=0.4)
ax5.axhline(y=125/1000, color='r', linestyle='--', linewidth = 0.5, alpha=0.4)
ax5.axhline(y=150/1000, color='r', linestyle='--', linewidth = 0.5, alpha=0.4) # 125
ax5.axhline(y=175/1000, color='r', linestyle='--', linewidth = 0.5, alpha=0.4)
ax5.axhline(y=200/1000, color='r', linestyle='--', linewidth = 0.5, alpha=0.4)
ax5.axhline(y=300/1000, color='r', linestyle='--', linewidth = 0.5, alpha=0.4) # 250
ax5.axhline(y=400/1000, color='r', linestyle='--', linewidth = 0.5, alpha=0.4)
ax5.axhline(y=500/1000, color='r', linestyle='--', linewidth = 0.5, alpha=0.4) # 500
ax5.axhline(y=600/1000, color='r', linestyle='--', linewidth = 0.5, alpha=0.4) # 500
"""
ax4.set_xlabel("Time [$\mu$s]")
ax4.set_ylabel("H [A/m]")
ax5.set_ylabel("dB/dt[T/s]")
ax4.legend()
ax4.grid(True), ax4.locator_params(axis='x', nbins=5)
ax4.set_ylim(-30, 30)

h1, l1 = ax4.get_legend_handles_labels()
h2, l2 = ax5.get_legend_handles_labels()
ax4.legend(h1+h2, l1+l2)

# BHループ
ax3 = fig.add_subplot(1, 2, 1) # 2行2列分割レイアウトの順序2にaxes追加
ax3.plot(H,   B, marker="None", linewidth = 0.5, label="$H$", color='orange')
ax3.plot(H_sim                  ,   B, marker="None", linewidth = 0.5, label="$H(Sim.)$", color='r')
ax3.plot(Hm,  B, marker="None", linewidth = 0.5, label="$H_m$", color='skyblue')
ax3.plot(saigen(table['Hm']),  real['B'], marker="None", linewidth = 0.5, label="$H_m(Sim.)$", color='b')
ax3.plot(Hhf, B, marker="None", linewidth = 0.5, label="$H_h+H_f$", color='pink')
ax3.plot(Hhf_sim,                   B, marker="None", linewidth = 0.5, label="$H_h+H_f(Sim.)$", color='purple')
ax3.grid(True), ax3.locator_params(axis='x', nbins=5)
ax3.set_xlim(-40, 40)
ax3.set_ylim(-0.25, 0.25)
ax3.set_xlabel("H [A/m]")
ax3.set_ylabel("B [T]")
ax3.legend()

"""
np.savetxt('BH.csv', np.vstack((B, H, H_sim )).transpose(),
           delimiter=',', \
           fmt="%.6f", \
           header="B, H, H_Sim", \
           comments="")
"""

# 共通
#fig.savefig("figure_{0:.4g}kHz.png".format(frequency))
plt.show()