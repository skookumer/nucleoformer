import numpy as np
from scipy.special import logsumexp
from pathlib import Path
from itertools import product
import random
from numba import njit
from collections import defaultdict

home = Path(__file__).parent


class Encoder_Decoder:

    def __init__(self, sequence_map=['A', 'C', 'G', 'T', 'N', '-']):
        self.sequence_map = sequence_map
        self.base_map = self.init_base_encoding_map()
        self.decode_map = np.array(sequence_map)
    
    def encode_seq(self, seq, k=0, pad=False):
        bytes = np.frombuffer("".join(seq).encode(), dtype=np.uint8)
        if k == 0:
            return np.array([self.base_map[bytes[i]] for i in range(len(bytes))], dtype=np.uint8)
        elif pad:
            pad = [5 for _ in range(k - 1)]
            lmerized = pad + [self.base_map[bytes[i]] for i in range(len(bytes))] + pad
            return np.array(lmerized, dtype=np.uint8)
        else:
            return np.array([self.base_map[bytes[i]] for i in range(len(bytes))], dtype=np.uint8)

    def decode_seq(self, seq: np.array):
        return "".join(self.decode_map[seq])

    def init_base_encoding_map(self):
        '''
        Returns mapping for binary/ASCII encoded strings to numbers
        '''
        base_map = np.zeros(256, dtype=np.uint8)
        for i in range(len(self.sequence_map)):
            base_map[ord(self.sequence_map[i])] = i
        return base_map


class HMM:

    def __init__(self, n_states, k, n_bases=6, model_name="model_a", encoder_decoder=None, fresh_start=False):
        self.n_states = n_states
        self.k = k
        self.dictionize(n_bases, k)
        self.vocab_size = len(self.vocab)

        if encoder_decoder == None:
            self.encoder_decoder = Encoder_Decoder()

        path = home / "models"
        path.mkdir(exist_ok=True, parents=True)
        self.matrix_path = path / f"{model_name}.npz"

        if self.matrix_path.exists() and not fresh_start:
            self.load_matrices()
        else:
            self.init_matrices()
    
    def dictionize(self, n_bases=6, k=3, avoid={4, 5}):
        self.vocab = {v: k for k, v in enumerate(product(range(n_bases), repeat=k))}
        self.vocab_inv = {k: v for k, v in enumerate(self.vocab)}
        self.prefix_lookup = defaultdict(list)
        for kmer, idx in self.vocab.items():
            if not any(b in avoid for b in kmer):
                self.prefix_lookup[kmer[:-1]].append(idx)
    
    def tokenize(self, seq, states=None):
        if isinstance(seq[0], str):
            seq = self.encoder_decoder.encode_seq(seq)
        win_size = self.k
        tokens = [self.vocab[tuple(seq[i:i+win_size])] for i in range(len(seq) - self.k + 1)]
        if states is None:
            return np.array(tokens, dtype=np.uint32)
        return np.array(tokens, dtype=np.uint32), states[self.k - 1:]


    def load_matrices(self):
        data = np.load(self.matrix_path)
        self.transition_matrix = data["transition_matrix"]
        self.emission_matrix = data["emission_matrix"]
        self.pi = data["pi"]
         
    def init_matrices(self):   
        N = self.n_states    
        self.transition_matrix = self.log_norm_random((N, N))
        self.emission_matrix = self.log_norm_random((N, self.vocab_size))
        self.pi = np.full(N, -np.log(N))
    
    def save_matrices(self):   
        np.savez(self.matrix_path,
                 transition_matrix=self.transition_matrix,
                 emission_matrix=self.emission_matrix,
                 pi=self.pi)
        
    def log_norm_random(self, shape, eps=.01):
            m = np.random.rand(*shape) + eps
            m /= m.sum(axis=-1, keepdims=True)
            return np.log(m)
    
    @staticmethod
    @njit(cache=True)
    def _viterbi_fast(seq, transition_matrix, emission_matrix, pi, T, N):
        V = np.full((N, T), -np.inf)
        BP = np.zeros((N, T), dtype=np.int64)

        #should I use initial probabilities here?
        V[:, 0] = pi + emission_matrix[:, seq[0]]     #initial probs + first observation

        for t in range(1, T):
            for s in range(N):
                scores = V[:, t - 1] + transition_matrix[:, s]
                best_score = np.argmax(scores)
                V[s, t] = scores[best_score] + emission_matrix[s, seq[t]]
                BP[s, t] = best_score
        
        best_path = np.zeros(T, dtype=np.int64)
        best_path[T-1] = np.argmax(V[:, T-1])
        
        
        for t in range(T-2, -1, -1):
            best_path[t] = BP[best_path[t+1], t+1]
        return best_path


    def viterbi(self, seq):
        T = len(seq)
        N = self.n_states
        return self._viterbi_fast(seq, self.transition_matrix, self.emission_matrix, self.pi, T, N)
    
    def gen_viterbi(self, seq, method="max_p"):

        def pyramid(n):
            half = np.clip(np.arange(1, n//2 + 2), a_min=1, a_max=self.k)
            if n % 2 == 0:
                return np.concatenate([half, half[::-1]])
            else:
                return np.concatenate([half, half[-2::-1]])

        states = self.viterbi(seq)

        if method == "max_p":
            T = len(states)
            weights = pyramid(T)
            token_probs = self.emission_matrix[states]
            best_indices = np.argmax(token_probs, axis=1)
            best_probs = token_probs[np.arange(T), best_indices]
            selection_matrix = np.zeros((T, len(weights)))
            i = 0
            for t in range(T):
                token = self.vocab_inv[best_indices[t]]
                for j in range(self.k):
                    selection_matrix[t, i + j] = token[j]
                i += 1
            output = []
            i = 0
            t = 0
            while i < len(weights):
                window_height = weights[t]
                values = selection_matrix[t:t+window_height, i]
                probs = best_probs[t:t+window_height]
                output.append(int(values[np.argmax(probs)]))
                i += 1
                if i > self.k:
                    t += 1
            return self.encoder_decoder.decode_seq(output)
    

    def prefixed_gen(self, prefix, current_kmer, current_state, prev_state):
        enc_pfx = tuple(int(v) for v in self.encoder_decoder.encode_seq(prefix))
        enc_kmer = tuple(int(v) for v in self.encoder_decoder.encode_seq(current_kmer))
        current_token_col = self.vocab[enc_kmer]

        if enc_pfx in self.prefix_lookup:
            em_cols = self.prefix_lookup[enc_pfx].copy()
        else:
            return None, None
        
        em_cols.remove(current_token_col)

        #em_cols contains indices
        em_sub = self.emission_matrix[:, em_cols].copy()                    #contains probabilities
        em_sub[current_state, :] = -np.inf
        em_scores = logsumexp(em_sub, axis=1)
        scores = self.transition_matrix[prev_state, :] + em_scores
        scores[current_state] = -np.inf
        probs = np.exp(scores - np.max(scores[scores != -np.inf]))
        probs /= probs.sum()
        new_state = np.random.choice(np.arange(self.n_states), p=probs)

        em_new = em_sub[new_state, :]                                       #contains probabilities
        probs = np.exp(em_new - np.max(em_new))
        probs /= probs.sum()
        new_token = em_cols[np.random.choice(len(em_cols), p=probs)]        #map probabilities back to indices
        kmer = np.array(self.vocab_inv[new_token])
        nucleotides = self.encoder_decoder.decode_seq(kmer)
        return nucleotides[-1], new_state
    
    def check_next_nucleotide(self, kmer, state):
        enc_kmer = tuple(int(v) for v in self.encoder_decoder.encode_seq(kmer))
        return self.emission_matrix[state, self.vocab[enc_kmer]] > -np.inf
    
    def fixed_state_gen(self, prefix, target_state):
        enc_pfx = tuple(int(v) for v in self.encoder_decoder.encode_seq(prefix))
        if enc_pfx in self.prefix_lookup:
            em_cols = self.prefix_lookup[enc_pfx].copy()
        else:
            raise ValueError("prefix failure")
        em_row = self.emission_matrix[target_state, em_cols]
        probs = np.exp(em_row - np.max(em_row))
        probs /= probs.sum()
        new_token = em_cols [np.random.choice(len(em_cols), p=probs)]
        kmer = np.array(self.vocab_inv[new_token])
        nucleotides = self.encoder_decoder.decode_seq(kmer)
        return nucleotides[-1]


    def baum_welch(self, seq, iterations=10):
        T = len(seq)

        #for i in range(iterations):
        if True:
            A = self.fwd(self.n_states, seq, self.pi, self.emission_matrix, self.transition_matrix)
            B = self.bwd(self.n_states, seq, self.emission_matrix, self.transition_matrix)
            p_x = logsumexp(A[:, -1])
            G = A + B - p_x

            xi = np.zeros((T - 1, self.n_states, self.n_states))
            for t in range(T - 1):
                xi[t] = (A[:, t].reshape(-1, 1) + self.transition_matrix + self.emission_matrix[:, seq[t+1]] + B[:, t+1] - p_x)

            self.pi = G[:, 0]
            self.transition_matrix = logsumexp(xi, axis=0) - logsumexp(G[:, :-1], axis=1).reshape(-1, 1)
            
            for k in range(self.vocab_size):
                mask = (seq == k)
                if np.any(mask):
                    self.emission_matrix[:, k] = logsumexp(G[:, mask], axis=1)
                
            self.emission_matrix -= logsumexp(self.emission_matrix, axis=1, keepdims=True)


    @staticmethod
    @njit(cache=False)
    def fwd(n_states, seq, pi, emission_matrix, transition_matrix):
        T = len(seq)
        A = np.full((n_states, T), -np.inf)
        A[:, 0] = pi + emission_matrix[:, seq[0]]     #initial probs + first observation

        for t in range(1, T):
            for s in range(n_states):
                c = A[:, t-1] + transition_matrix[:, s]
                max_a = np.max(c)
                prev = max_a + np.log(np.sum(np.exp(c - max_a)))
                A[s, t] = prev + emission_matrix[s, seq[t]]
        return A
    
    @staticmethod
    @njit(cache=False)
    def bwd(n_states, seq, emission_matrix, transition_matrix):
        T = len(seq)
        B = np.full((n_states, T), -np.inf)
        B[:, T-1] = 0

        for t in range(T-2, -1, -1):
            for s in range(n_states):
                c = transition_matrix[s, :] + emission_matrix[:, seq[t + 1]] + B[:, t+1]
                max_a = np.max(c)
                B[s, t] = max_a + np.log(np.sum(np.exp(c - max_a)))
        return B
        
    @staticmethod
    @njit(cache=False)
    def make_xi(n_states, T, A, B, p_x, seq, emission_matrix, transition_matrix):
        xi = np.zeros((T - 1, n_states, n_states))
        for t in range(T - 1):
            x = seq[t+1]
            a_t = A[t].reshape(-1, 1)
            em = emission_matrix[x].reshape(1, -1) #right reshape?
            b_t = B[t+1]
            xi[t] = a_t + transition_matrix + em + b_t - p_x
            # xi[t] = A[t].reshape(-1, 1) + transition_matrix + emission_matrix[seq[t+1]].reshape(1, -1) + B[:, t+1] - p_x
        return xi
    
    def baum_welch_E_batch(self, seqs):
        return [self.baum_welch_E(self.tokenize(seq)) for seq in seqs]
            
    def baum_welch_E(self, seq):
        T = len(seq)
        A = self.fwd(self.n_states, seq, self.pi, self.emission_matrix, self.transition_matrix)
        B = self.bwd(self.n_states, seq, self.emission_matrix, self.transition_matrix)
        p_x = logsumexp(A[:, -1])
        G = A + B - p_x
        xi = self.make_xi(self.n_states, T, 
                          np.ascontiguousarray(A.T), 
                          np.ascontiguousarray(B.T), 
                          p_x, seq, 
                          np.ascontiguousarray(self.emission_matrix.T), 
                          self.transition_matrix)
        return G, xi, p_x
    
    def posterior_decode(self, seq):
        T = len(seq)
        A = self.fwd(self.n_states, seq, self.pi, self.emission_matrix, self.transition_matrix)
        B = self.bwd(self.n_states, seq, self.emission_matrix, self.transition_matrix)
        p_x = logsumexp(A[:, -1])
        G = A + B - p_x
        return np.argmax(G, axis=0), p_x
    
    def ll_only(self, seq):
        A = self.fwd(self.n_states, seq, self.pi, self.emission_matrix, self.transition_matrix)
        p_x = logsumexp(A[:, -1])
        return p_x

    def baum_welch_M(self, G, xi, seqs, smooth=-700):
        if isinstance(G, tuple):
            xi_sum = np.full((self.n_states, self.n_states), -np.inf)
            G_sum = np.full(self.n_states, -np.inf)

            for g, x in zip(G, xi):
                T = x.shape[0]
                log_T = np.log(T)
                xi_sum = np.logaddexp(xi_sum, logsumexp(x, axis=0) - log_T)
                G_sum = np.logaddexp(G_sum, logsumexp(g[:, :-1], axis=1) - log_T)

            self.transition_matrix = xi_sum - G_sum.reshape(-1, 1)
        else:
            self.transition_matrix = (logsumexp(xi, axis=0) - logsumexp(G[:, :-1], axis=1).reshape(-1, 1))

        em_num = np.full((self.n_states, self.vocab_size), -np.inf)
        for k in range(self.vocab_size):
            mask = seqs == k
            if np.any(mask):
                em_num[:, k] = logsumexp(G[:, mask], axis=1)

        em_num = np.where(em_num == -np.inf, smooth, em_num)
        self.emission_matrix = em_num - logsumexp(em_num, axis=1, keepdims=True)

    
    def MLE(self, seq, states):
        T = len(seq)
        A = self.count_transitions(T, self.n_states, states)
        self.transition_matrix = A - logsumexp(A, axis=1, keepdims=True)
        E = self.count_emissions(T, self.n_states, self.vocab_size, states, seq)
        self.emission_matrix = E - logsumexp(E, axis=1, keepdims=True)
    
    def MLE_E(self, seq, states):
        if len(seq) != len(states):
            raise ValueError("len seq, states does not match")
        T = len(seq)
        A = self.count_transitions(T, self.n_states, states)
        E = self. count_emissions(T, self.n_states, self.vocab_size, states, seq)
        return A, E
    
    def MLE_M(self, A, E):
        if type(A) == list:
            A = np.sum(A, axis=0)
            E = np.sum(E, axis=0)
        self.transition_matrix = self.ln_norm(A)
        self.emission_matrix = self.ln_norm(E)
    
    @staticmethod
    def ln_norm(M):
        ln_sums = logsumexp(M, axis=1, keepdims=True)
        out = M - ln_sums
        out[np.isneginf(ln_sums.squeeze())] = -np.inf
        return out

    @staticmethod
    @njit(cache=False)
    def count_transitions(T, n_states, states):
        A = np.full((n_states, n_states), 0)
        for t in range(T - 1):
            i, j = states[t], states[t+1]
            A[i, j] += 1
        return A
    
    @staticmethod
    @njit(cache=False)
    def count_emissions(T, n_states, vocab_size, states, seq):
        E = np.full((n_states, vocab_size), 0)
        for t in range(T):
            s, k = states[t], seq[t]
            E[s, k] += 1
        return E
        


    def train_model_BW(self, seqs, epochs=3, toprint=False):

        for i in range(epochs):
            if toprint:
                print(f"Epoch {i + 1}")
            for j, seq in enumerate(seqs):
                if toprint:
                    print(f"Iteration {j}", end="\r", flush=True)
                enc_seq = self.encoder_decoder.encode_seq(seq, k=self.k)
                tok_seq = self.tokenize(enc_seq)
                self.baum_welch(tok_seq)
            self.save_matrices()



def generate_genomic_sequences(reference_seq, num_sequences, mutation_rate=0.1):
    """
    Generate a 2D list of genomic sequences based on a reference sequence
    
    Args:
        reference_seq (str): Base sequence to mutate from
        num_sequences (int): Number of variant sequences to generate
        mutation_rate (float): Probability of mutation per position (0.0 to 1.0)
    
    Returns:
        list: 2D list where each row is a sequence (as list of nucleotides)
    """
    nucleotides = ['A', 'T', 'G', 'C']
    sequences = []
    
    # Add the reference sequence as first entry
    sequences.append(list(reference_seq))
    
    # Generate mutated variants
    for _ in range(num_sequences - 1):
        variant = []
        for nucleotide in reference_seq:
            if random.random() < mutation_rate:
                # Mutate: pick a different nucleotide
                possible = [n for n in nucleotides if n != nucleotide]
                variant.append(random.choice(possible))
            else:
                # Keep original
                variant.append(nucleotide)
        sequences.append(variant)
    
    return sequences




if __name__ == "__main__":

    reference = "ATGCGATCGTAGCTAGCTAG"
    sequences = generate_genomic_sequences(reference, num_sequences=5, mutation_rate=0.15)

    model = HMM(n_states=5, k=3, n_bases=6, model_name="testmodel", fresh_start=False)
    # model.train_model(sequences)
    # print(model.emission_matrix)
    # print(model.transition_matrix)
    # print(model.pi)
    seq = model.encoder_decoder.encode_seq(reference, k=0)
    seq = model.tokenize(seq)
    x = model.gen_viterbi(seq)
    print(x)




