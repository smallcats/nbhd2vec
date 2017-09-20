import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine

import pickle

nbhd_ct_table = pd.read_msgpack('nbhd_ct_table')

zero_cols = []
for k in nbhd_ct_table.columns:
    if sum(nbhd_ct_table[k]) == 0: zero_cols.append(k)

nbhd_ct_table = nbhd_ct_table.drop(zero_cols, axis=1)

#tf-idf with entropy
gf = nbhd_ct_table.sum().values
N = nbhd_ct_table.values
p = N*(1/gf)

good_log = np.vectorize(lambda x: np.log(x) if x>0 else 0)

g = 1-sum(p*good_log(p)/np.log(422))
A = g*np.log(N+1)

#SVD
U,s,V = np.linalg.svd(A)
s_rel = s/sum(s)

for k in range(100):
    M_rand = np.std(M)*np.random.randn(*M.shape)+np.mean(M)
    U_rand, s_rand, V_rand = np.linalg.svd(M_rand)
    s_randrel = s_rand/sum(s_rand)
    pct = [sum(s_randrel[:j]) for j in range(len(s))]
    for n, p in enumerate(pct):
        if k == 0: accum = pct
        if k > 0: accum[n] = (accum[n]*k + pct[n])/(k+1)

max_improvement = max(enumerate([sum(s_rel[:k])-accum[k] for k in range(len(s))]),key=lambda x: x[1])[0]

div_idx = 0
for nbhd in nbhd_labels:
    if nbhd[0] != 'New York': break
    div_idx += 1

nyc_vec = np.mean(U[:div_idx, :], axis=0)
phl_vec = np.mean(U[div_idx:, :], axis=0)

U_minus_city = U
for k in range(div_idx):
    U_minus_city[k, :] -= nyc_vec
for k in range(div_idx,U.shape[0]):
    U_minus_city[k,:] -= phl_vec

np.save('tfidf_mtx.npy', A)
np.save('U.npy', U)
np.save('Ucity.npy',U_minus_city)
np.save('s.npy', s)
np.save('V.npy', V)
with open('nbhd_labels.p', 'wb') as file:
	pickle.dump(list(nbhd_ct_table.index), file)
np.save('ndims.npy', np.array([max_improvement]))