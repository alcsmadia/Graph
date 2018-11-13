import sys #ドラッグドロップ
import matplotlib.pyplot as plt # プロット
import numpy as np # 数学ライブラリ
import os
from mpl_toolkits.mplot3d import Axes3D
import subprocess

for j in range(1, len(sys.argv)):
    # --------- ファイルを読む ---------
    data = np.genfromtxt(sys.argv[j], delimiter=",", skip_header=2, dtype='float',
                         names=["Time", "Ch1", "Ch2"])

    # ----------------- 定数 -------------------
    def parameter(x): # ファイル名から変数を読む関数
       return(os.path.basename(sys.argv[j]).split(" ")[x]) # csv名を空白で区切ったx番目
    frequency   = float(parameter(1).replace("kHz", ""))
    dbdt        = int(parameter(3).replace("(", "").replace("mTµs)", ""))
    N           = parameter(5).replace("n", "-").split("-")
    N           = [N[0], int(N[1]), int(N[2])]
    renji       = float(parameter(6).replace("range.csv", "" ))

    T           = 1 / (frequency * 10**3) * 10**6 # 周期(μs
    point_all   = len(data) # 列数
    dt          = renji / point_all
    point_float = T / dt # 1周期ポイント数=全ポイント数/レンジ(個/秒)*周期(秒)
    point       = int(round(point_float)) # 丸めてからint型に

    DCCT_late_time = 6*10**(-3) # μs
    DCCT_late = round(point / T * DCCT_late_time)

    jirotyo, jiromenseki = 0.031852683, 0.00001207 # エクセルでは外径と内径から計算している
    #jirotyo, jiromenseki = 0.0635, 0.0000654

    # --------- 読み込んだデータフレームの加工 ---------
    Current_late  = data['Ch1'][int(DCCT_late):] # Ch1の列をDCCTの遅れ分ずらす
    begin         = np.nanargmin(np.abs(data['Ch2'][0:int(point_all/4)]))
    CurrentT      = Current_late[begin:begin+point] # 1周期の開始点からポイント数行まで取り出す
    VoltageT      = data['Ch2'][begin:begin+point]
    Time          = data['Time'][begin:begin+point] * 10**6 #1周期の絶対時間(μs)
    TimeT         = Time - Time[0] #1周期の相対時間
    t = np.arange(0, dt * point, dt * point / 1600) # 1/4周期で平均をとるためにポイント数を4の倍数にする
    amari = len(t) % 4
    t = t[:len(t)-amari]

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
            an    = (2 / T) * i_cos.sum() * dt
            i_m  += an       * fcos(t    , n)
            
            i_sin = i_data * fsin(t_data, n)
            bn    = (2 / T) * i_sin.sum() * dt
            i_hf += bn      * fsin(t    , n)
            
        return [i_m, i_hf]
    
    i_m, i_hf = fourier(CurrentT, TimeT)[0],  fourier(CurrentT, TimeT)[1]
    v         = fourier(VoltageT, TimeT)[0] + fourier(VoltageT, TimeT)[1]
    
    # --------- BHループ計算の準備 ---------
    i             = i_m + i_hf
    H             = i * N[1] / jirotyo # Hl=Ni
    Hm            = i_m * N[1] / jirotyo
    if j == 1:
        Hh        = i_hf * N[1] / jirotyo
    Hf            = i_hf * N[1] / jirotyo
    int_v_dt      = np.cumsum(v * dt)
    B             = int_v_dt / (N[2] * jiromenseki) * 10**-6 #NBA=∫vdt
    B_fix         = B-(B.max() + B.min())/2
    
    B_begin       = int(np.nanargmin(np.abs(B_fix[:800])))
    B_fix         = np.concatenate((B_fix[B_begin:], B_fix[:B_begin]))
    H             = np.concatenate((H[B_begin:]    ,  H[:B_begin]))
    Hm            = np.concatenate((Hm[B_begin:]   , Hm[:B_begin]))
    if j == 1:
        Hh        = np.concatenate((Hh[B_begin:]   , Hh[:B_begin]))
    Hf            = np.concatenate((Hf[B_begin:]   , Hf[:B_begin])) - Hh
    
    # --------- T/4で折りたたむ ---------
    Hm1, Hm2, Hm3, Hm4 = np.split(Hm, 4)
    Hm  = (Hm\
           + np.concatenate((Hm2[::-1], -1 * Hm3[::-1], Hm4[::-1], -1 * Hm1[::-1]))\
           - 1 * np.concatenate((Hm3, Hm4, Hm1, Hm2))\
           + np.concatenate((-1 * Hm4[::-1], Hm1[::-1], -1 * Hm2[::-1], Hm3[::-1]))\
           )/4
    
    if j == 1:
        Hh1, Hh2, Hh3, Hh4 = np.split(Hh, 4)
        Hh = (np.concatenate((Hh1, Hh2, Hh3, Hh4))\
            - np.concatenate((Hh2[::-1], -1 * Hh3[::-1], Hh4[::-1], -1 * Hh1[::-1]))\
             - 1 * np.concatenate((Hh3, Hh4, Hh1, Hh2))\
            + np.concatenate((Hh4[::-1], -1 * Hh1[::-1], Hh2[::-1], -1 * Hh3[::-1]))\
               )/4
    
    Hf1, Hf2, Hf3, Hf4 = np.split(Hf, 4)
    Hf = (np.concatenate((Hf1, Hf2, Hf3, Hf4))\
        - np.concatenate((Hf2[::-1], -1 * Hf3[::-1], Hf4[::-1], -1 * Hf1[::-1]))\
         - 1 * np.concatenate((Hf3, Hf4, Hf1, Hf2))\
        + np.concatenate((Hf4[::-1], -1 * Hf1[::-1], Hf2[::-1], -1 * Hf3[::-1]))\
           )/4
    
    H  = Hm + Hh + Hf
    
    B_fix1, B_fix2, B_fix3, B_fix4 = np.split(B_fix, 4)
    B_fix    = (B_fix\
           + np.concatenate((B_fix2[::-1], -1 * B_fix3[::-1], B_fix4[::-1], -1 * B_fix1[::-1]))\
           - 1 * np.concatenate((B_fix3, B_fix4, B_fix1, B_fix2))\
           + np.concatenate((-1 * B_fix4[::-1], B_fix1[::-1], -1 * B_fix2[::-1], B_fix3[::-1]))\
           )/4
    """
    # --------- 描写 -----------
    fig = plt.figure(figsize=(10,8)) # グラフを表示する(5,4), (24,4)
    fig.suptitle("dB/dt={0} | Bm={1:.3g} mT | {2} | turns{3}:{4}".format(parameter(3), B_fix.max()*1000, parameter(2) ,N[1], N[2]))
    fig.subplots_adjust(hspace=0.3, wspace=0.5)

    ax1 = fig.add_subplot(2, 2, 1) # 2行2列分割レイアウトの順序1にaxes追加
    ax2 = ax1.twinx()  # ax2をax1に関連付ける
    ax1.set_title("↓フーリエ級数展開後↓", fontdict={'family': 'IPAexGothic'})
    ax1.plot(t, i, marker="None", label="Current", color='b', linewidth = 0.5)
    #ax1.plot(TimeT[center-begin], CurrentT[center-begin], marker="o")
    ax2.plot(t, v, marker="None", label="Voltage", color='r', linewidth = 0.5)
    #ax1.legend(bbox_to_anchor=(0.1, 1.15), loc='upper left')
    #ax2.legend(bbox_to_anchor=(0.5, 1.15), loc='upper left')
    ax1.grid(True), ax1.locator_params(axis='x', nbins=5)
    ax1.set_xlabel("Time [$\mu$s]")
    ax1.set_ylabel("Current [A]"), ax2.set_ylabel("Voltage [V]")

    # BHループ
    ax3 = fig.add_subplot(2, 2, 2) # 2行2列分割レイアウトの順序2にaxes追加
    ax3.plot(H,  B_fix, marker="None", linewidth = 0.5, label="$H$")
    ax3.plot(Hm, B_fix, marker="None", linewidth = 0.5, label="$H_m$")
    ax3.plot(Hh, B_fix, marker="None", linewidth = 0.5, label="$H_h$")
    ax3.plot(Hf, B_fix, marker="None", linewidth = 0.5, label="$H_f$")
    ax3.grid(True), ax3.locator_params(axis='x', nbins=5)
    ax3.set_xlabel("H(Magnetic field intensity) [A/m]")
    ax3.set_ylabel("B(Magnetic flux density) [T]")
    ax3.legend()

    # iの成分
    ax4 = fig.add_subplot(2, 2, 3)
    ax4.plot(t, i_m, marker="None", label="$i_m$", color='b', linewidth = 0.5)
    ax4.plot(t, i_hf, marker="None", label="$i_h+i_f$", color='skyblue', linewidth = 0.5)
    ax4.set_xlabel("Time [$\mu$s]")
    ax4.set_ylabel("Current [A]")
    ax4.grid(True), ax4.locator_params(axis='x', nbins=5)
    ax4.legend()
    
    # 共通
    fig.savefig("figure_{0:.4g}kHz.png".format(frequency))

    #plt.show()
    
    """
    if j == 1:
        X = Y = Z = CSV = np.empty(0)
    for q in range(0, len(t)):
        X = np.append(X, dbdt)
    Y = np.append(Y, B_fix)
    Z = np.append(Z, Hm) # Hm
    if j == len(sys.argv)-1:
        print(len(X))
        print(len(Y))
        print(len(Z))
        CSV = np.vstack((X, Y, Z))
        CSV = CSV.transpose()
        np.savetxt('out.csv', CSV, delimiter=',', \
        fmt    = ["%.4f", "%.4f", "%.4f"], \
        header = "FormatType,102,\nmemo1,,\nmemo2,,", \
        footer = "[ScaleX]\nLabel=dB/dt\n[ScaleY]\nLabel=Hm\n[ScaleZ]\nLabel=B", \
        comments="")
        subprocess.Popen([r'C:/Users/Hidenori/OneDrive/ドキュメント/プログラミング/PycharmProjects/Research/sub/GraphR242/GraphR.exe', \
                          r'C:/Users/Hidenori/OneDrive/ドキュメント/プログラミング/PycharmProjects/Research/out.csv'])