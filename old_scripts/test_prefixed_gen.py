from HMM import HMM
import numpy as np

def test_prefixed_gen():
    hmm = HMM(n_states=9, k=6, n_bases=6, model_name="supervised_10k_oldschema")
    state_sums = np.exp(hmm.emission_matrix).sum(axis=1)
    current_state = np.argmax(state_sums)
    current_token = np.argmax(hmm.emission_matrix[current_state, :])
    sample_kmer = hmm.vocab_inv[current_token]

    # pick a real prefix and kmer from the vocab
    prev_state = 0

    decoded_kmer = hmm.encoder_decoder.decode_seq(np.array(sample_kmer))
    decoded_prefix = decoded_kmer[:-1]
    import time
    t0 = time.time()
    nucleotide, new_state = hmm.prefixed_gen(decoded_prefix, decoded_kmer, current_state, prev_state)
    print(time.time() - t0)
    print(sample_kmer)
    print(nucleotide)

    # # assert nucleotide in hmm.encoder_decoder, f"unexpected nucleotide: {nucleotide}"
    # assert new_state != current_state, "state should change"
    # assert 0 <= new_state < hmm.n_states, f"state out of range: {new_state}"

    new_kmer = decoded_prefix + nucleotide
    assert new_kmer != decoded_kmer, "k-mer should change"

    print(f"nucleotide: {nucleotide}, new_state: {new_state}, new_kmer: {new_kmer}")

test_prefixed_gen()

test_prefixed_gen()