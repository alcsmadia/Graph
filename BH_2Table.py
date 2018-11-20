from BH_1Calculate import *

Hf_t = np.concatenate((np.empty(800), Hf_t))

# 観賞用のテーブル
# def save(name, csv):
#     np.savetxt("table_{0}.csv".format(name), csv, delimiter=',', \
#                fmt="%.6f", \
#                header="FormatType,102,\nmemo1,,\nmemo2,,", \
#                footer="[GraphContour]\nSurfaceType=0\n[GraphPoint]\nPointType=2\n\
#                   [ScaleX]\nLabel=dBdt\n[ScaleY]\nLabel=B\n[ScaleZ]\nLabel={0}".format(name), \
#                comments="")
#
# lam_f = X[800:] / Hf_t
# CSVf = np.vstack((X[800:] * 1000, Y[800:] * 1000, lam_f)).transpose()
# CSVm = np.vstack((X * 1000      , Y * 1000       , Hm_t)).transpose()
# save("Hm", CSVm), save("λf", CSVf)

# 実際に使うテーブル
np.savetxt('Table.csv'.format(j), np.vstack((X, Y, Hm_t, Hh_t, Hf_t, Hhf_t)).transpose(),
           delimiter=',', \
           fmt="%.6f", \
           header="dBdt,B,Hm,Hh,Hf,Hhf", \
           comments="")

input("Success.")