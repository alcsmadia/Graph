import sys #ドラッグドロップ
#import matplotlib
#matplotlib.use('Agg') # -----(1)
import matplotlib.pyplot as plt # プロット
import numpy as np # 数学ライブラリ
from scipy import integrate
import os

for j in range(1, len(sys.argv)):
    # --------- ファイルを読む ---------
    data = np.genfromtxt(sys.argv[j], delimiter=",", skip_header=2, dtype='float',
                         names=["Time", "Ch1", "Ch2"])

    # ----------------- 定数 -------------------
    def parameter(x): # ファイル名から変数を読む関数
       return(os.path.basename(sys.argv[j]).split(" ")[x]) # csv名を空白で区切ったx番目
    frequency   = float(parameter(1).replace("kHz", "" ))
    N           = parameter(5).replace("n", "-").split("-")
    N           = [N[0], int(N[1]), int(N[2])]
    renji       = float(parameter(6).replace("range.csv", "" ))

    T           = 1 / (frequency * 10**3) * 10**6 # 周期(μs）
    point_all   = len(data) # 列数
    dt          = renji / point_all
    point_float = T / dt # 1周期ポイント数=全ポイント数/レンジ(個/秒)*周期(秒)
    point       = int(round(point_float)) # 丸めてからint型に

    DCCT_late_time = 6*10**(-3) # μs
    DCCT_late = round(point / T * DCCT_late_time)

    jirotyo, jiromenseki = 0.031852683, 0.00001207 # エクセルでは外径と内径から計算している

    # --------- 読み込んだデータフレームの加工 ---------
    Current_late  = data['Ch1'][int(DCCT_late):] # Ch1の列をDCCTの遅れ分ずらす
    center        = np.nanargmax(Current_late) + round(0.001*T/dt) # 最大値を探して少し調整
    begin         = int(center - (point / 2)) # 1周期の開始点 np.nanargmin(np.abs(data['Ch2'][0:400]))
    CurrentT      = Current_late[begin:begin+point] # 1周期の開始点からポイント数行まで取り出す
    Voltage_point = data['Ch2'][begin:begin+point]
    VoltageT      = Voltage_point-Voltage_point.mean()
    Time          = data['Time'][begin:begin+point] * 10**6 #1周期の絶対時間(μs)
    TimeT         = Time - Time[0] #1周期の相対時間
    t = np.arange(0, dt * point, dt)

    # --------- フーリエ級数展開 ---------
    def fourier(i_data, t_data):
        fourier = 161 #フーリエ級数展開の回数
        
        def fcos(t, n):
            return np.cos(n * 2 * np.pi / T * t) # cos(n * ωt)
        def fsin(t, n):
            return np.sin(n * 2 * np.pi / T * t)

        i_m = i_hf = 0
        for n in range(1, fourier + 1):
            i_cos = i_data * fcos(t_data, n)
            an    = integrate.cumtrapz((2 / T)  * i_cos, t_data)[point-2]
            i_m  += an       * fcos(t    , n)
            
            i_sin = i_data * fsin(t_data, n)
            bn    = integrate.cumtrapz((2 / T)  * i_sin, t_data)[point-2]
            i_hf += bn      * fsin(t    , n)
        return [i_m, i_hf]
    
    i_m, i_hf = fourier(CurrentT, TimeT)[0], fourier(CurrentT, TimeT)[1]
    i         = i_m + i_hf
    v         = fourier(VoltageT, TimeT)[0] + fourier(VoltageT, TimeT)[1]
    
    # --------- BHループ計算の準備 ---------
    H             = i * N[1] / jirotyo # Hl=Ni
    #H             = H[:-1] # 配列数をint_v_dtと合わせる
    int_v_dt      = integrate.cumtrapz(v, t)
    int_v_dt      = np.append(int_v_dt, (v[-1]+v[0])*dt/2)
    B             = int_v_dt / (N[2] * jiromenseki) * 10**-6 #NBA=∫vdt
    B_fix         = B-(B.max() + B.min())/2

    # --------- 描写 -----------
    fig = plt.figure(figsize=(7.5,6)) # グラフを表示する(5,4), (24,4)
    fig.suptitle("dB/dt={0} | Bm={1:.3g} mT | {2} | turns{3}:{4}".format(parameter(3), B_fix.max()*1000, parameter(2) ,N[1], N[2]))

    ax1 = fig.add_subplot(2, 2, 1) # 2行2列分割レイアウトの順序1にaxes追加
    ax2 = ax1.twinx()  # ax2をax1に関連付ける
    ax1.plot(t, i, marker="None", label="Current", color='b', linewidth = 0.5)
    ax2.plot(t, v, marker="None", label="Voltage", color='r', linewidth = 0.5)
    #ax1.legend(bbox_to_anchor=(0.1, 1.15), loc='upper left')
    #ax2.legend(bbox_to_anchor=(0.5, 1.15), loc='upper left')
    ax1.grid(True)
    ax1.locator_params(axis='x', nbins=5)
    ax1.set_ylabel("Current [A]")
    ax1.set_xlabel("Time [$\mu$s]")
    ax2.set_ylabel("Voltage [V]")

    # BHループ
    ax3 = fig.add_subplot(2, 2, 2) # 2行2列分割レイアウトの順序2にaxes追加
    ax3.plot(H, B_fix, marker="None", linewidth = 0.5)
    ax3.grid(True)
    ax3.locator_params(axis='x', nbins=5)
    ax3.set_ylabel("B(Magnetic flux density) [T]")
    ax3.set_xlabel("H(Magnetic field intensity) [A/m]")

    # iの成分
    ax4 = fig.add_subplot(2, 2, 3)
    ax4.plot(t, i_m, marker="None", label="$i_m$", color='b', linewidth = 0.5)
    ax4.plot(t, i_hf, marker="None", label="$i_h+i_f$", color='skyblue', linewidth = 0.5)
    ax4.set_ylabel("Current [A]")
    ax4.set_xlabel("Time [$\mu$s]")
    ax4.grid(True)
    ax4.locator_params(axis='x', nbins=5)
    ax4.legend()

    # 共通
    plt.subplots_adjust(hspace=0.3, wspace=0.5)
    #plt.savefig("figure_{0:.4g}kHz.png".format(frequency))

plt.show()
