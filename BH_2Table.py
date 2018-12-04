from BH_1Calculate import *

# 観賞用のテーブル --------------------------------------------
Hf_t    = np.concatenate((np.full(half, np.nan), Hf_t)) # Hfが存在しない範囲にNaNを入れる
lam_f = X / Hf_t

def save(name, csv):
     np.savetxt("table_{0}.csv".format(name), csv, delimiter=',', \
                fmt="%.3f", \
                header="FormatType,102,\nmemo1,,\nmemo2,,", \
                footer="[GraphContour]\nSurfaceType=0\n[GraphPoint]\nPointType=2\n[ScaleX]\nLabel=dBdt\n[ScaleY]\nLabel=B\n[ScaleZ]\nLabel={0}".format(name), \
                comments="")

#------------------------------------------
notNaN = np.where(X == X) # dbdtがNaNでないインデックスを取得
X09 = X[notNaN] # 0.9以下を除いた配列をつくる
Y09 = Y[notNaN]
Bm09  = Bm[notNaN]
lam_f09 = lam_f[notNaN]
Hm_t09 = Hm_t[notNaN]

CSVm = np.vstack((X09 * 1000, Y09 * 1000, Bm09 * 1000, Hm_t09)).transpose()

#------------------------------------------
not0 = np.where(lam_f == lam_f) # Hfが存在する範囲のインデックスを取得
X09 = X[not0]
Y09 = Y[not0]
Bm09  = Bm[not0]
lam_f09 = lam_f[not0]
Hm_t09 = Hm_t[notNaN]

CSVf = np.vstack((X09 * 1000, Y09 * 1000, Bm09 * 1000, lam_f09 * 10**6)).transpose() # mT/μs と mT と T*m/A*s
save("Hm", CSVm), save("λf", CSVf)

# 実際に使うテーブル ------------------------------------------
np.savetxt('Table.csv'.format(j), np.vstack((X, Y, Bm, Hm_t, Hh_t, Hf_t, Hhf_t)).transpose(),
           delimiter=',', \
           fmt="%.6f", \
           header="dBdt,B,Bm,Hm,Hh,Hf,Hhf", \
           comments="")

input("Success.")