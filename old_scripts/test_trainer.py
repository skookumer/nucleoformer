from aligner import GeneLoader
from HMM import HMM
import numpy as np


l = GeneLoader("hg38_primary")

hmm_sv = HMM(n_states=len(l.states), k=6, n_bases=6, model_name="supervised_10k_oldschema", fresh_start=True)

A = np.ones((hmm_sv.n_states, hmm_sv.n_states))
E = np.ones((hmm_sv.n_states, hmm_sv.vocab_size))
for i in range(1, 10000):
    entry = l[i]
    seq, states = hmm_sv.tokenize(entry["seq"], entry["states"])
    A_sub, E_sub = hmm_sv.MLE_E(seq, states)
    A += A_sub
    E += E_sub
    print(f"iteration {i}        ", end='\r', flush=True)
hmm_sv.MLE_M(A, E)
hmm_sv.save_matrices()

