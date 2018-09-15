import sys #ドラッグドロップ
import pandas as pd # csvを開く
#import matplotlib
#matplotlib.use('Agg') # -----(1)
import matplotlib.pyplot as plt # プロット
import numpy as np # 数学ライブラリ
from scipy import integrate
import os

for j in range(1, len(sys.argv)):
    # --------- ファイルをpandasで読む ---------
    data = pd.read_csv(sys.argv[j], engine='python', header=1)
    data.columns=["Time", "Ch1", "Ch2"] # 読み込んだファイルに列名をつける

    # ----------------- 定数 -------------------
    def parameter(x): # ファイル名から変数を読む関数
       return(os.path.basename(sys.argv[j]).split(" ")[x]) # csv名を空白で区切ったx番目
    frequency   = float(parameter(1).replace("kHz", "" ))
    N           = parameter(5).replace("n", "-").split("-")
    N           = [N[0], int(N[1]), int(N[2])]
    renji       = float(parameter(6).replace("range.csv", "" ))
    
    #frequency   = 1000
    #N           = [0, 3, 1]
    #renji       = 1.2

    T           = 1 / (frequency * 10**3) * 10**6 # 周期(μs）
    point_all   = len(data) # 列数
    dt          = renji / point_all
    point_float = T / dt # 1周期ポイント数=全ポイント数/レンジ(個/秒)*周期(秒)
    point       = int(round(point_float)) # 丸めてからint型に

    DCCT_late_time = 6*10**(-3) # μs
    DCCT_late = round(point / T * DCCT_late_time)

    jirotyo = 0.031852683 # エクセルでは外径と内径から計算している
    jiromenseki = 0.00001207

    # 読み込んだデータフレームの加工
    Current_late  = data['Ch1'].shift(-int(DCCT_late)) # Ch1の列をDCCTの遅れ分ずらす
    #begin = int(Current_late.iloc[500:1500].idxmax() - (point / 2)) # 1周期の開始点を探してint型にする
    begin = (np.abs(data['Ch2'].iloc[0:400])).idxmin() # 0～200の絶対値で電圧が0に近い点（絶対ポイント）
    begin -= int(round(2.3773*np.log(frequency) - 5))
    CurrentT = Current_late.iloc[begin:begin+point] # 1周期の開始点からポイント数行まで取り出す
    Voltage_point = data['Ch2'].iloc[begin:begin+point]
    VoltageT      = Voltage_point-Voltage_point.mean()
    Time          = (data['Time'].iloc[begin:begin+point])*10**6 #1周期の絶対時間(μs)
    TimeT         = Time - Time.iloc[0] #1周期の相対時間
    t          = np.arange(0, dt * point, dt)

    # --------- フーリエ級数展開 ---------
    i_m = i_hf = v = 0
    fourier = 161 #int(input("フーリエ級数展開の回数："))
    
    def fcos(t, n):
        return np.cos(2 * np.pi * n * t / T)
    def fsin(t, n):
        return np.sin(2 * np.pi * n * t / T)
    
    for n in range(1, fourier + 1):
        i_cos = CurrentT * fcos(TimeT, n)
        an    = integrate.simps((2 / T)  * i_cos, TimeT)
        i_m  += an       * fcos(t    , n)
        
        i_sin = CurrentT * fsin(TimeT, n)
        bn    = integrate.simps((2 / T)  * i_sin, TimeT)
        i_hf += bn      * fsin(t    , n)
        
        v_cos = VoltageT * fcos(TimeT, n)
        an_V  = integrate.simps((2 / T)  * v_cos, TimeT)
        v_sin = VoltageT * fsin(TimeT, n)
        bn_V  = integrate.simps((2 / T)  * v_sin, TimeT)
        v    += an_V     * fcos(t    , n) +\
                bn_V     * fsin(t    , n)
    i = i_m + i_hf
    
    # --------- BHループ計算の準備 ---------
    H             = i * N[1] / jirotyo # Hl=Ni
    H             = H[:-1] # 配列数をint_v_dtと合わせる
    int_v_dt      = integrate.cumtrapz(v, t) * 10**-6 #∫vdt
    B             = int_v_dt / (N[2] * jiromenseki) #NBA=∫vdt
    B_fix         = B-(B.max() + B.min())/2

    print("Bm                                                                     %f"     % B_fix.max())
    #a, b = np.polyfit(TimeT.iloc[begin:v_half], B_fix.iloc[begin:v_half], 1) # 線形回帰
    #print("dB/dt  %f"     % a)
    # print("理想：%s | 実際：%f mT/μs" % (parameter(3), round(a*1000, 3)))
    #print("%s | turns %d：%d" % (parameter(3), N[1], N[2]))
    print(integrate.simps(H, B))

    # --------- 描写 -----------
    fig = plt.figure(figsize=(5,4)) # グラフを表示する(5,4), (24,4)
    fig.suptitle("dB/dt={0} | Bm={1:.3g} mT | {2} | turns{3}:{4}".format(parameter(3), B_fix.max()*1000, parameter(2) ,N[1], N[2]))

    ax1 = fig.add_subplot(2, 2, 1) # 2行2列分割レイアウトの順序1にaxes追加
    ax2 = ax1.twinx()  # ax2をax1に関連付ける
    ax1.plot(t, i, marker="None", label="Current", color='b', linewidth = 0.5)
    ax2.plot(t, v, marker="None", label="Voltage", color='r', linewidth = 0.5)
    ax1.legend(bbox_to_anchor=(0.1, 1.15), loc='upper left')
    ax2.legend(bbox_to_anchor=(0.5, 1.15), loc='upper left')
    ax1.grid(True)
    ax1.locator_params(axis='x', nbins=5)
    ax1.set_ylabel("Current [A]")
    ax1.set_xlabel("Time [$\mu$s]")
    ax2.set_ylabel("Voltage [V]")

    # BHループ
    ax3 = fig.add_subplot(2, 2, 2) # 2行2列分割レイアウトの順序2にaxes追加
    ax3.plot(H, B_fix, marker="None", linewidth = 0.5)
    # ax3.set_xlim([-60,60])
    # ax3.set_ylim([-0.5,0.5])
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

    # 空き
    ax5 = fig.add_subplot(2, 2, 4)
    #ax5.plot(TimeT.iloc[v_half-30:v_half], B_fix.iloc[v_half-30:v_half], marker="o")
    ax5.axis('off')

    # 共通
    plt.subplots_adjust(hspace=0.2, wspace=0.5)
    #plt.savefig("figure_{1:.5g}kHz.png".format(frequency))

plt.show()
