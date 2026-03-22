from aligner import GeneLoader
from HMM import HMM
import time


l = GeneLoader("hg38_primary")
hmm_sv = HMM(n_states=len(l.states), k=6, n_bases=6, model_name="sv_a", fresh_start=True)
# hmm_bm = HMM(n_states=len(l.states) * 2, k=6)

t0 = time.time()
for i in range(1, 11):
    seq, conf, states = l.get_idx(i)
    seq, states = hmm_sv.tokenize(seq, states)
    A, E = hmm_sv.MLE_E(seq, states)
    hmm_sv.MLE_M(A, E)
    print(f"iteration {i}, time {time.time() - t0}")
    t0 = time.time()
    pred = hmm_sv.viterbi(seq)
    print(states[:20])
    print(pred[:20])
    input()
