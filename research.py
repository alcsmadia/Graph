import sys #ドラッグドロップ
import pandas as pd # csvを開く
import matplotlib.pyplot as plt # プロット
import numpy as np # 数学ライブラリ
import os

# --------- ファイルをpandasで読む ---------
data = pd.read_csv(sys.argv[1], engine='python', header=1)
#data = pd.read_csv('C:\\Users\\Hidenori\\Desktop\\2121.csv', engine='python', header=1)
data.columns=["Time", "Ch1", "Ch2"] # 読み込んだファイルに列名をつける

# ----------------- 定数 -------------------
def parameter(x): # ファイル名から変数を読む関数
    return(os.path.basename(sys.argv[1]).split(" ")[x]) # csv名を空白で区切ったx番目
frequency   = float(parameter(1).replace("kHz", "" ))
N           = parameter(5).replace("n", "-").split("-")
N           = [N[0], int(N[1]), int(N[2])]
renji       = float(parameter(6).replace("range.csv", "" ))

# frequency = 1125
# N         = [0, 3, 1]
# renji     = 1.08

T           = 1 / (frequency * 10**3) * 10**6 # 周期(μs）
point_all   = len(data) # 列数
dt          = renji / point_all
point_float = T / dt # 1周期ポイント数=全ポイント数/レンジ(個/秒)*周期(秒)
point       = int(round(point_float)) # 丸めてからint型に

DCCT_late_time = 6*10**(-3) # μs
DCCT_late = round(point * DCCT_late_time / T)

jirotyo = 0.031852683 # エクセルでは外径と内径から計算している
jiromenseki = 0.00001207

# 読み込んだデータフレームの加工
Current_late  = data['Ch1'].shift(-int(DCCT_late)) # Ch1の列をDCCTの遅れ分ずらす
begin = int(Current_late.iloc[500:1500].idxmax() - (point / 2)) # 1周期の開始点を探してint型にする
Current_point = Current_late.iloc[begin:begin+point] # 1周期の開始点からポイント数行まで取り出す
CurrentT      = Current_point - Current_point.mean() # 縦のゼロ合わせ
Voltage_point = data['Ch2'].iloc[begin:begin+point]
VoltageT      = Voltage_point-Voltage_point.mean()
Time          = (data['Time'].iloc[begin:begin+point])*10**6 #1周期の絶対時間(μs)
TimeT         = Time - Time.iloc[0] #1周期の相対時間


# --------- フーリエ級数展開 ---------
def tenkai_cos(x, n):  # 関数の定義
    cos    = np.cos(2 * np.pi * n * TimeT / T)
    x_cos  = x * cos
    an     = (2 / T) * x_cos.sum() * dt
    an_cos = an * cos
    return (an_cos)

def tenkai_sin(x, n):
    sin    = np.sin(2 * np.pi * n * TimeT / T)
    x_sin  = x * sin
    bn     = (2 / T) * x_sin.sum() * dt
    bn_sin = bn * sin
    return (bn_sin)

i_m = i_hf = v = 0
fourier = 161 #int(input("フーリエ級数展開の回数："))
for n in range(1, fourier + 1):
    i_m  += tenkai_cos(CurrentT, n)
    i_hf += tenkai_sin(CurrentT, n)
    v += tenkai_cos(VoltageT, n) + tenkai_sin(VoltageT, n)
i = i_m + i_hf

# --------- BHループ計算の準備 ---------
v_0           = (np.abs(v.iloc[0:200])).idxmin() # 0～200の絶対値で電圧が0に近い点（絶対ポイント）
v_dt          = v.iloc[v_0-begin:1000] * dt #（相対ポイント）
int_v_dt      = np.cumsum(v_dt) # 累積和をとる=∫vdt
v_half        = (np.abs(int_v_dt.iloc[0:1000] - int_v_dt/2)).idxmin() #0～1000で電圧がint_v_dt/2に近い点
v_BH          = v.iloc[v_half-begin: ].append(v.iloc[ :v_half-begin]) # BHループ用v, v_half以前と以後の並び替え
i_BH          = i.iloc[v_half-begin: ].append(i.iloc[ :v_half-begin]) # BHループ用i

# BHループ計算
int_v_BH_dt   = np.cumsum(v_BH * dt * 10**-6) # 累積和をとる
H             = i_BH * N[1] / jirotyo # Hl=Ni
B             = int_v_BH_dt / (N[2] * jiromenseki) #NBA=∫vdt
B_fix         = B-(B.max() + B.min())/2

print("周波数 %f kHz" % frequency)
print("巻数   %d：%d" % (N[1], N[2]))
print("レンジ %f"     % renji)
print("Bm                                                                     %f"     % B_fix.max())
a, b = np.polyfit(TimeT.iloc[v_half-30:v_half], B_fix.iloc[v_half-30:v_half], 1) # 線形回帰
print("dB/dt  %f"     % a)

# --------- 描写 -----------
fig = plt.figure(figsize=(5,4)) # グラフを表示する
ax1 = fig.add_subplot(2, 2, 1) # 2行2列分割レイアウトの順序1にaxes追加
ax2 = ax1.twinx()  # ax2をax1に関連付ける
ax1.plot(data['Time'], data['Ch1'], marker="None", label="Current", color='b')
ax2.plot(data['Time'], data['Ch2'], marker="None", label="Voltage", color='r')
ax1.grid(True)
ax1.locator_params(axis='x', nbins=5)
ax1.set_ylabel("Current [A]")
ax1.set_xlabel("Time [s]")
ax2.set_ylabel("Voltage [V]")
ax1.legend(bbox_to_anchor=(0.1, 1.15), loc='upper left')
ax2.legend(bbox_to_anchor=(0.5, 1.15), loc='upper left')

# BHループ
ax3 = fig.add_subplot(2, 2, 2) # 2行2列分割レイアウトの順序2にaxes追加
ax3.plot(H, B_fix, marker="None")
ax3.grid(True)
ax3.locator_params(axis='x', nbins=5)
ax3.set_ylabel("B(Magnetic flux density) [T]")
ax3.set_xlabel("H(Magnetic field intensity) [A/m]")

# iの成分
ax4 = fig.add_subplot(2, 2, 3)
ax4.plot(TimeT-1, i_m, marker="None", label="$i_m$", color='b')
ax4.plot(TimeT-1, i_hf, marker="None", label="$i_h+i_f$", color='skyblue')
ax4.grid(True)
ax4.locator_params(axis='x', nbins=5)
ax4.set_ylabel("Current [A]")
ax4.set_xlabel("Time [$\mu$s]")
ax4.legend()

# 空き
ax5 = fig.add_subplot(2, 2, 4)
#ax5.plot(TimeT.iloc[v_half-30:v_half], B_fix.iloc[v_half-30:v_half], marker="o")
ax5.axis('off')

# 共通
plt.subplots_adjust(hspace=0.2, wspace=0.5)
plt.show()
