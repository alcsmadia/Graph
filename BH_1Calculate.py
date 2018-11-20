import sys #ドラッグドロップ
import numpy as np # 数学ライブラリ
import os
import matplotlib.pyplot as plt # プロット

# --------- フーリエ級数展開 ---------
def fourier(i_data, t_data):
    fourier = 161  # フーリエ級数展開の回数

    def fcos(t, n):
        return np.cos(n * 2 * np.pi / T * t)  # cos(n * ωt)
    def fsin(t, n):
        return np.sin(n * 2 * np.pi / T * t)

    i_m = i_hf = 0
    for n in range(1, fourier + 1):
        i_cos = i_data * fcos(t_data, n)
        an = (2 / T) * i_cos.sum() * dt
        i_m += an * fcos(t, n)

        i_sin = i_data * fsin(t_data, n)
        bn = (2 / T) * i_sin.sum() * dt
        i_hf += bn * fsin(t, n)
    return [i_m, i_hf]

# ---------------- T/4で折りたたむ ---------
def T4(B, B_begin, gu_ki):
    B = np.concatenate((B[B_begin:], B[:B_begin]))
    B1, B2, B3, B4 = np.split(B, 4)

    if gu_ki == "gukansu":
        B0 = (B1 + B2[::-1] - B3 - B4[::-1]) / 4
        B = np.concatenate((B0, B0[::-1], -1 * B0, -1 * B0[::-1]))
    else:
        B0 = (B1 - B2[::-1] - B3 + B4[::-1]) / 4
        B = np.concatenate((B0, -1 * B0[::-1], -1 * B0, B0[::-1]))
    return (B)

def parameter(j, x): # ファイル名から変数を読む関数
    return (os.path.basename(sys.argv[j]).split(" ")[x])  # csv名を空白で区切ったx番目

def junbi(j):
    # ---------- ファイル名から定数を設定する ---------
    data = np.genfromtxt(sys.argv[j], delimiter=",", skip_header=2, dtype='float',
                         names=["Time", "Ch1", "Ch2"])

    frequency = float(parameter(j, 1).replace("kHz", ""))
    N = parameter(j, 5).replace("n", "-").split("-")
    N = [N[0], int(N[1]), int(N[2])]
    renji = float(parameter(j, 6).replace("range.csv", ""))

    T = 1 / (frequency * 10 ** 3) * 10 ** 6  # 周期(μs）
    point_all = len(data)  # 列数
    dt = renji / point_all
    point_float = T / dt  # 1周期ポイント数=全ポイント数/レンジ(個/秒)*周期(秒)
    point = int(round(point_float))  # 丸めてからint型に

    DCCT_late_time = 6 * 10 ** (-3)  # μs
    DCCT_late = round(point / T * DCCT_late_time)

    # -------------- 読み込んだデータの加工 ---------
    Current_late = data['Ch1'][int(DCCT_late):]  # Ch1の列をDCCTの遅れ分ずらす
    begin = np.nanargmin(np.abs(data['Ch2'][0:int(point_all / 4)]))
    CurrentT = Current_late[begin:begin + point]  # 1周期の開始点からポイント数行まで取り出す
    VoltageT = data['Ch2'][begin:begin + point]
    Time = data['Time'][begin:begin + point] * 10 ** 6  # 1周期の絶対時間(μs)
    TimeT = Time - Time[0]  # 1周期の相対時間
    dt2 = dt * point / 1600  # 1/4周期で平均をとるためにポイント数を4の倍数にする
    t = np.arange(0, dt * point, dt2)  # 1/4周期で平均をとるためにポイント数を4の倍数にする
    amari = len(t) % 4
    t = t[:len(t) - amari]  # なんか1600点にならないときがあるので
    return CurrentT, VoltageT, TimeT, t, T, point, N, dt, dt2

def keisan_B(VoltageT, TimeT):
    v = fourier(VoltageT, TimeT)[0] + fourier(VoltageT, TimeT)[1]
    int_v_dt = np.cumsum(v * dt2)
    B = int_v_dt / (N[2] * jiromenseki) * 10 ** -6  # NBA=∫vdt
    B = B - (B.max() + B.min()) / 2
    B_begin = int(np.nanargmin(np.abs(B[:800])))
    B = T4(B, B_begin, "gukansu")
    return v, B, B_begin

def keisan_H(CurrentT, TimeT):
    i_m, i_hf = fourier(CurrentT, TimeT)[0], fourier(CurrentT, TimeT)[1]
    i_m, i_hf = T4(i_m, B_begin, "gukansu"), T4(i_hf, B_begin, "kikansu")
    i = i_m + i_hf
    H = i * N[1] / jirotyo  # Hl=Ni
    Hm = i_m * N[1] / jirotyo
    Hhf = i_hf * N[1] / jirotyo
    return i, i_m, i_hf, H, Hm, Hhf

def hozon():
    B_pre = np.concatenate((B[1:], B[:1]))  # Bを1ポイントずらす
    dbdt = (B_pre - B) / dt2

    # ------------------ 結果の保存 -----------------
    def exchange(B): # 半周期ぶん取り出す（つなげる）
        B1, B2, B3, B4 = np.split(B, 4)
        return(np.concatenate((B4, B1)))
    np.savetxt('real.csv', np.vstack((exchange(dbdt), exchange(B), exchange(H), \
                                      exchange(Hm), exchange(Hhf))).transpose(), delimiter=',', \
               fmt="%.6f", \
               header="dBdt,B,H,Hm,Hhf", \
               comments="")
    np.savetxt('t_and_N.csv', np.vstack((t, np.append(X, np.full(len(t), N[1])))).transpose(), delimiter=',', \
               fmt="%.6f", \
               header="t,N", \
               comments="")

def graph():
    fig = plt.figure(figsize=(10, 8))  # グラフを表示する(5,4), (24,4)
    fig.suptitle(
        "dB/dt={0} | Bm={1:.3g} mT | {2} | turns{3}:{4}".format(parameter(j, 3), B.max() * 1000, parameter(j, 2), N[1],
                                                                N[2]))
    fig.subplots_adjust(hspace=0.3, wspace=0.5)

    ax1 = fig.add_subplot(2, 2, 1)  # 2行2列分割レイアウトの順序1にaxes追加
    ax2 = ax1.twinx()  # ax2をax1に関連付ける
    ax1.set_title("↓フーリエ級数展開後↓", fontdict={'family': 'IPAexGothic'})
    ax1.plot(t, i, marker="None", label="Current", color='b', linewidth=0.5)
    ax2.plot(t, v, marker="None", label="Voltage", color='r', linewidth=0.5)
    ax1.legend(bbox_to_anchor=(0.1, 1.25), loc='upper left')
    ax2.legend(bbox_to_anchor=(0.5, 1.25), loc='upper left')
    ax1.grid(True), ax1.locator_params(axis='x', nbins=5)
    ax1.set_xlabel("Time [$\mu$s]")
    ax1.set_ylabel("Current [A]"), ax2.set_ylabel("Voltage [V]")

    Hhf1, Hhf2, Hhf3, Hhf4 = np.split(Hhf, 4)
    B1, B2, B3, B4 = np.split(B, 4)
    # BHループ
    ax3 = fig.add_subplot(2, 2, 2)  # 2行2列分割レイアウトの順序2にaxes追加
    ax3.plot(H, B, marker="None", linewidth=0.5, label="$H$")
    ax3.plot(Hm, B, marker="None", linewidth=0.5, label="$H_m$")
    ax3.plot(Hh, B, marker="None", linewidth = 0.5, color='#d62728', label="$H_h$")
    if j != 1:
        ax3.plot(Hf, B, marker="None", linewidth = 0.5, label="$H_f$")
    #ax3.plot(Hhf, B, marker="None", linewidth=0.5, label="$H_h+H_f$")
    ax3.grid(True), ax3.locator_params(axis='x', nbins=5)
    ax3.set_xlabel("H(Magnetic field intensity) [A/m]")
    ax3.set_ylabel("B(Magnetic flux density) [T]")
    ax3.legend()

    # iの成分
    ax4 = fig.add_subplot(2, 2, 3)
    ax4.plot(t, i_m, marker="None", label="$i_m$", color='b', linewidth=0.5)
    ax4.plot(t, i_hf, marker="None", label="$i_h+i_f$", color='skyblue', linewidth=0.5)
    ax4.set_xlabel("Time [$\mu$s]")
    ax4.set_ylabel("Current [A]")
    ax4.grid(True), ax4.locator_params(axis='x', nbins=5)
    ax4.legend()

    # 共通
    # fig.savefig("figure_{0:.4g}kHz.png".format(frequency))
    plt.show()

# --------------- テーブル作成 ----------------
def table():
    global X, Y, Hm_t, Hh_t, Hf_t, Hhf_t
    B_pre = np.concatenate((B[1:], B[:1]))  # Bを前に1ポイントずらした
    dbdt = (B_pre - B) / dt2

    def exchange(B): # 半周期ぶん取り出す（つなげる）
        B1, B2, B3, B4 = np.split(B, 4)
        return(np.concatenate((B4, B1)))

    dbdt1 = int(parameter(j, 3).replace("(", "").replace("mTµs)", "")) / 1000
    X = np.append(X, exchange(dbdt)) # X = np.append(X, np.full(800, dbdt1)) #
    Y = np.append(Y, exchange(B))
    Hm_t = np.append(Hm_t, exchange(Hm))
    Hh_t = np.append(Hh_t, exchange(Hh))
    Hhf_t = np.append(Hhf_t, exchange(Hhf))
    if j != 1:
        Hf_t = np.append(Hf_t, exchange(Hf))

# ------------------ 本体 -----------------
jirotyo, jiromenseki = 0.031852683, 0.00001207  # エクセルでは外径と内径から計算している
# jirotyo, jiromenseki = 0.0635, 0.0000654

X = Y = Hm_t = Hhf_t = Hh_t = Hf_t = np.empty(0)
for j in range(1, len(sys.argv)):
    CurrentT, VoltageT, TimeT, t, T, point, N, dt, dt2 = junbi(j)
    v, B, B_begin = keisan_B(VoltageT, TimeT)
    i, i_m, i_hf, H, Hm, Hhf = keisan_H(CurrentT, TimeT)
    if j == 1: Hh = Hhf
    Hf = Hhf - Hh
    if __name__ == '__main__':
        hozon()
        #graph()
    if __name__ == 'BH_1Calculate':
        table()