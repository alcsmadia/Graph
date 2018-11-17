import sys #ドラッグドロップ
import matplotlib.pyplot as plt # プロット
import numpy as np # 数学ライブラリ
import os

X = Y = Hm_t = Hh_t = Hf_t = np.empty(0)
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

    T           = 1 / (frequency * 10**3) * 10**6 # 周期(μs）
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
    dt2            = dt * point / 1600 # 1/4周期で平均をとるためにポイント数を4の倍数にする
    t             = np.arange(0, dt * point, dt2)
    amari = len(t) - 1600
    t = t[:len(t)-amari] # なんか1600点にならないときがあるので

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
    int_v_dt      = np.cumsum(v * dt2)
    B             = int_v_dt / (N[2] * jiromenseki) * 10**-6 #NBA=∫vdt
    B_fix         = B-(B.max() + B.min())/2
    
    B_begin       = int(np.nanargmin(np.abs(B_fix[:800])))
    B_fix         = np.concatenate((B_fix[B_begin:], B_fix[:B_begin]))
    v             = np.concatenate((v[B_begin:]    , v[:B_begin]))
    
    # ---------------- T/4で折りたたむ ---------
    B_fix1, B_fix2, B_fix3, B_fix4 = np.split(B_fix, 4)
    B_fix    = (B_fix\
           + np.concatenate((B_fix2[::-1], -1 * B_fix3[::-1], B_fix4[::-1], -1 * B_fix1[::-1]))\
           - 1 * np.concatenate((B_fix3, B_fix4, B_fix1, B_fix2))\
           + np.concatenate((-1 * B_fix4[::-1], B_fix1[::-1], -1 * B_fix2[::-1], B_fix3[::-1]))\
           )/4
    
    i_m     = np.concatenate((i_m[B_begin:] , i_m[:B_begin]))
    i_hf    = np.concatenate((i_hf[B_begin:], i_hf[:B_begin]))
    
    im1, im2, im3, im4 = np.split(i_m, 4)
    i_m  = (i_m\
           + np.concatenate((im2[::-1], -1 * im3[::-1], im4[::-1], -1 * im1[::-1]))\
           - 1 * np.concatenate((im3, im4, im1, im2))\
           + np.concatenate((-1 * im4[::-1], im1[::-1], -1 * im2[::-1], im3[::-1]))\
           )/4
    
    ihf1, ihf2, ihf3, ihf4 = np.split(i_hf, 4)
    i_hf = (i_hf\
        - np.concatenate((ihf2[::-1], -1 * ihf3[::-1], ihf4[::-1], -1 * ihf1[::-1]))\
         - 1 * np.concatenate((ihf3, ihf4, ihf1, ihf2))\
        + np.concatenate((ihf4[::-1], -1 * ihf1[::-1], ihf2[::-1], -1 * ihf3[::-1]))\
           )/4
    
    i             = i_m + i_hf
    H             = i * N[1] / jirotyo # Hl=Ni
    Hm            = i_m * N[1] / jirotyo
    if j == 1:
        Hh        = i_hf * N[1] / jirotyo
    Hf            = i_hf * N[1] / jirotyo - Hh
    
    # --------------- テーブル作成 ----------------
    B_pre = np.concatenate((B_fix[1:] , B_fix[:1])) # Bを1ポイントずらす
    dbdt  = (B_pre - B_fix) / dt2
    
    H1, H2, H3, H4 = np.split(H, 4)
    Hm1, Hm2, Hm3, Hm4 = np.split(Hm, 4)
    Hh1, Hh2, Hh3, Hh4 = np.split(Hh, 4)
    Hf1, Hf2, Hf3, Hf4 = np.split(Hf, 4)
    B_fix1, B_fix2, B_fix3, B_fix4 = np.split(B_fix, 4)
    dbdt1, dbdt2, dbdt3, dbdt4 = np.split(dbdt, 4)
    dbdt = np.concatenate((dbdt4, dbdt1))
    
    X = np.append(X, dbdt) # X = np.append(X, np.full(800, dbdt)
    Y = np.append(Y, np.concatenate((B_fix4, B_fix1)))
    Hm_t = np.append(Hm_t, np.concatenate((Hm4, Hm1)))
    Hh_t = np.append(Hh_t, np.concatenate((Hh4, Hh1)))
    if j != 1:
        Hf_t  = np.append(Hf_t, np.concatenate((Hf4, Hf1)))

lam_f = X[800:] / Hf_t
Hf_t  = np.concatenate((np.empty(800), Hf_t))

def save(name, csv):
    np.savetxt("table_{0}.csv".format(name), csv, delimiter=',', \
    fmt    = "%.6f", \
    header = "FormatType,102,\nmemo1,,\nmemo2,,", \
    footer = "[GraphContour]\nSurfaceType=0\n[GraphPoint]\nPointType=2\n\
              [ScaleX]\nLabel=dBdt\n[ScaleY]\nLabel=B\n[ScaleZ]\nLabel={0}".format(name), \
    comments="")
    
CSVm  = np.vstack((X*1000      , Y*1000      , Hm_t)).transpose()
CSVf  = np.vstack((X[800:]*1000, Y[800:]*1000, lam_f)).transpose()

save("Hm" , CSVm)
save("λf", CSVf)

    # --------------- 再現 ----------------
test = np.genfromtxt("C:\\Users\\Hidenori\\Desktop\\data2.csv", delimiter=",", dtype='float', names=["x", "y"])

def saigen(Z):
    x0 = y0 = z0 = np.empty(0)
    for ii in range(0, len(test)):
        data_dbdt   = np.array([5, 10, 20, 30, 40, 50, 60, 80, 100, 125, 150,\
                                175, 200, 300, 400, 500, 600, 700, 800, 900], dtype=object)
        data_dbdt = data_dbdt/1000
        x           = test['x'][ii] #float(input("dbdt:"))
        dbdt_close  = (np.abs(data_dbdt - x)).argsort()[:2] # xに近い順にdbdtを2点取りだす
        dbdt_close  = np.sort(dbdt_close) # 取り出したdbdtを昇順にする
        
        y           = test['y'][ii] #float(input("B   :"))
        # yに近い順にBを2*2点取り出す
        B_close1    = (np.abs(Y[dbdt_close[0] * 800 : (dbdt_close[0] + 1) * 800] - y)).argsort()[:2] + dbdt_close[0] * 800
        B_close2    = (np.abs(Y[dbdt_close[1] * 800 : (dbdt_close[1] + 1) * 800] - y)).argsort()[:2] + dbdt_close[1] * 800
        B_close1, B_close2 = np.sort(B_close1), np.sort(B_close2)
        
        x1, y1, z1  = X[B_close1[0]], Y[B_close1[0]], Z[B_close1[0]]
        x2, y2, z2  = X[B_close1[1]], Y[B_close1[1]], Z[B_close1[1]]
        x3, y3, z3  = X[B_close2[0]], Y[B_close2[0]], Z[B_close2[0]]
        x4, y4, z4  = X[B_close2[1]], Y[B_close2[1]], Z[B_close2[1]]
        
        if y > (y4-y1)/(x4-x1) * (x-x1) + y1:
            z=((x1-x)*((y2-y1)*(z4-z1)-(y4-y1)*(z2-z1))+(y1-y)*((z2-z1)*(x4-x1)-(z4-z1)*(x2-x1)))/((x2-x1)*(y4-y1)-(x4-x1)*(y2-y1))+z1
        else:
            z=((x1-x)*((y2-y1)*(z3-z1)-(y3-y1)*(z2-z1))+(y1-y)*((z2-z1)*(x3-x1)-(z3-z1)*(x2-x1)))/((x2-x1)*(y3-y1)-(x3-x1)*(y2-y1))+z1
        z0 = np.append(z0, z)
    return (z0)

X_T = np.concatenate((test['x'], -1 * test['x']))
Y_T = np.concatenate((test['y'], -1 * test['y']))
H_T = saigen(Hm_t) + saigen(Hh_t) + saigen(Hf_t)
H_T = np.concatenate((H_T      , -1 * H_T))

np.savetxt('saigen_Hm.csv', np.vstack((test['x'], test['y'], saigen(Hm_t))).transpose(), delimiter=',')
np.savetxt('saigen_Hh.csv', np.vstack((test['x'], test['y'], saigen(Hh_t))).transpose(),  delimiter=',')
np.savetxt('saigen_Hf.csv', np.vstack((test['x'], test['y'], saigen(Hf_t))).transpose(), delimiter=',')
np.savetxt('saigen_H.csv' , np.vstack((X_T, Y_T, H_T)).transpose(), delimiter=',')