import numpy as np
import sys
from os.path import join as pjoin


arg_folder = sys.argv[1]
mc = np.loadtxt(pjoin(arg_folder, 'man.test.ec.txt'))
wc = np.loadtxt(pjoin(arg_folder, 'man_wit.test.ec.txt'))
wac = np.loadtxt(pjoin(arg_folder, 'man_wit.test.avg.ec.txt'))
mw = np.loadtxt(pjoin(arg_folder, 'man.test.ew.txt'))
ww = np.loadtxt(pjoin(arg_folder, 'man_wit.test.ew.txt'))
waw = np.loadtxt(pjoin(arg_folder, 'man_wit.test.avg.ew.txt'))
num_man = mc.shape[0]
num_wit = wc.shape[0]
total = num_man + num_wit
w1 = num_man * 1. / total
w2 = num_wit * 1. / total
macro_lc = (np.sum(mc[:,0]/mc[:,-1]) + np.sum(wc[:,0]/wc[:,-1])) / total
macro_c = (np.sum(mc[:,1]/mc[:,-1]) + np.sum(wc[:,1]/wc[:,-1])) / total
macro_lw = (np.sum(mw[:,0]/mw[:,-1]) + np.sum(ww[:,0]/ww[:,-1]))/ total
macro_w = (np.sum(mw[:,1]/mw[:,-1]) + np.sum(ww[:,1]/ww[:,-1])) / total
macro_lca = np.mean(mc[:,0]/mc[:,-1]) * w1 + np.mean(wac[:,0]/wac[:,-1])  * w2
macro_ca = np.mean(mc[:,1]/mc[:,-1]) * w1 + np.mean(wac[:,1]/wac[:,-1]) * w2
macro_lwa = np.mean(mw[:,0]/mw[:,-1]) * w1 + np.mean(waw[:,0]/waw[:,-1]) * w2
macro_wa = np.mean(mw[:,1]/mw[:,-1]) * w1 + np.mean(waw[:,1]/waw[:,-1]) * w2
#print np.mean(mw[:,1]/mw[:,-1]), np.mean(waw[:,1]/waw[:,-1]), macro_wa
#print (np.sum(mw[:,1]/mw[:,-1]) + np.sum(waw[:,1]/waw[:,-1]))/total
micro_lc = (np.sum(mc[:,0]) + np.sum(wc[:,0]))/(np.sum(mc[:,-1]) + np.sum(wc[:, -1]))
micro_c = (np.sum(mc[:, 1]) + np.sum(wc[:,1]))/(np.sum(mc[:,-1]) + np.sum(wc[:, -1]))
micro_lw = (np.sum(mw[:,0]) + np.sum(ww[:,0]))/(np.sum(mw[:,-1]) + np.sum(ww[:,-1]))
micro_w=(np.sum(mw[:,1]) + np.sum(ww[:,1]))/ (np.sum(mw[:,-1]) + np.sum(ww[:,-1]))
micro_lca = (np.sum(mc[:,0]) + np.sum(wac[:,0]))/(np.sum(mc[:,-1]) + np.sum(wac[:,-1]))
micro_ca = (np.sum(mc[:, 1]) + np.sum(wac[:,1])) / (np.sum(mc[:, -1]) + np.sum(wac[:,-1]))
micro_lwa = (np.sum(mw[:, 0]) + np.sum(waw[:,0])) / (np.sum(mw[:,-1])  + np.sum(waw[:,-1]))
micro_wa = (np.sum(mw[:, 1]) + np.sum(waw[:,1])) / (np.sum(mw[:,-1]) + np.sum(waw[:,-1]))
#print w1, w2
print macro_c, micro_c, macro_lc, micro_lc, macro_w, micro_w, macro_lw, micro_lw
print macro_ca, micro_ca, macro_lca, micro_lca, macro_wa, micro_wa, macro_lwa, micro_lwa
