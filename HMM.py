import numpy as np
from scipy.special import logsumexp
from pathlib import Path
from itertools import product
import random
from numba import njit

home = Path(__file__).parent


class Encoder_Decoder:

    def __init__(self, sequence_map=['A', 'C', 'G', 'T', 'N', '-']):
        self.sequence_map = sequence_map
        self.base_map = self.init_base_encoding_map()
        self.decode_map = np.array(sequence_map)
    
    def encode_seq(self, seq, k=0, pad=False):
        bytes = np.frombuffer("".join(seq).encode(), dtype=np.uint8)
        if k == 0:
            return np.array([self.base_map[bytes[i]] for i in range(len(bytes))])
        elif pad:
            pad = [5 for _ in range(k - 1)]
            lmerized = pad + [self.base_map[bytes[i]] for i in range(len(bytes))] + pad
            return np.array(lmerized)
        else:
            return np.array([self.base_map[bytes[i]] for i in range(len(bytes))])

    def decode_seq(self, seq):
        return "".join(self.decode_map[seq])

    def init_base_encoding_map(self):
        '''
        Returns mapping for binary/ASCII (not sure) encoded strings to numbers
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
    
    def dictionize(self, n_bases=6, k=3):
        self.vocab = {v: k for k, v in enumerate(product(range(n_bases), repeat=k))}
        self.vocab_inv = {k: v for k, v in enumerate(self.vocab)}
    
    def tokenize(self, seq, states=None):
        if isinstance(seq[0], str):
            seq = self.encoder_decoder.encode_seq(seq)
        win_size = self.k
        tokens = [self.vocab[tuple(seq[i:i+win_size])] for i in range(len(seq) - self.k + 1)]
        if states is None:
            return np.array(tokens)
        return np.array(tokens), states[self.k - 1:]


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

    def viterbi(self, seq):
        
        T = len(seq)
        N = self.n_states

        V = np.full((N, T), -np.inf)
        BP = np.zeros((N, T), dtype=int)

        #should I use initial probabilities here?
        V[:, 0] = self.pi + self.emission_matrix[:, seq[0]]     #initial probs + first observation

        for t in range(1, T):
            for s in range(N):
                scores = V[:, t - 1] + self.transition_matrix[:, s]
                best_score = np.argmax(scores)
                V[s, t] = scores[best_score] + self.emission_matrix[s, seq[t]]
                BP[s, t] = best_score
        
        best_path = np.zeros(T, dtype=int)
        best_path[T-1] = np.argmax(V[:, T-1])
        
        
        for t in range(T-2, -1, -1):
            best_path[t] = BP[best_path[t+1], t+1]
        
        return best_path
    
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
            

    def fwd(self, seq):
        T = len(seq)
        A = np.full((self.n_states, T), -np.inf)
        A[:, 0] = self.pi + self.emission_matrix[:, seq[0]]     #initial probs + first observation

        for t in range(1, T):
            for s in range(self.n_states):
                prev = logsumexp(A[:, t-1] + self.transition_matrix[:, s])
                A[s, t] = prev + self.emission_matrix[s, seq[t]]
        return A
    
    def bwd(self, seq):
        T = len(seq)
        B = np.full((self.n_states, T), -np.inf)
        B[:, T-1] = 0

        for t in range(T-2, -1, -1):
            for s in range(self.n_states):
                B[s, t] = logsumexp(self.transition_matrix[s, :] + self.emission_matrix[:, seq[t + 1]] + B[:, t+1])
        return B
    
    def baum_welch(self, seq, iterations=10):
        T = len(seq)

        #for i in range(iterations):
        if True:
            A = self.fwd(seq)
            B = self.bwd(seq)
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
    
    def baum_welch_E(self, seq):
        T = len(seq)
        A = self.fwd(seq)
        B = self.bwd(seq)
        p_x = logsumexp(A[:, -1])
        G = A + B - p_x

        xi = np.zeros((T - 1, self.n_states, self.n_states))
        for t in range(T - 1):
            xi[t] = (A[:, t].reshape(-1, 1) + self.transition_matrix + 
                    self.emission_matrix[:, seq[t+1]] + B[:, t+1] - p_x)
        
        return G, xi, p_x
    
    def baum_welch_M(self, G, xi, seqs):

        self.transition_matrix = (logsumexp(xi, axis=0) - logsumexp(G[:, :-1], axis=1).reshape(-1, 1))

        for k in range(self.vocab_size):
            mask = np.concatenate([seq == k for seq in seqs])
            if np.any(mask):
                self.emission_matrix[:, k] = logsumexp(G[:, mask], axis=1)
        
        self.emission_matrix -= logsumexp(self.emission_matrix, axis=1, keepdims=True)
    
    def MLE(self, seq, states):
        T = len(seq)
        A = self.count_transitions(T, self.n_states, states)
        self.transition_matrix = A - logsumexp(A, axis=1, keepdims=True)
        E = self.count_emissions(T, self.n_states, self.vocab_size, states, seq)
        self.emission_matrix = E - logsumexp(E, axis=1, keepdims=True)

        #probably not necessary
        # self.pi = np.full(self.n_states, -np.inf)
        # self.pi[states[0]] = 0.0
    

    def MLE_E(self, seq, states):
        if len(seq) != len(states):
            raise ValueError("len seq, states does not match")
        T = len(seq)
        A = self.count_transitions(T, self.n_states, states)
        E = self. count_emissions(T, self.n_states, self.vocab_size, states, seq)
        return A, E
    
    def MLE_M(self, A, E):
        if type(A) == list:
            A = np.array(A)
            E = np.array(E)
        self.transition_matrix = self.ln_norm(A)
        self.emission_matrix = self.ln_norm(E)
    
    @staticmethod
    def ln_norm(M):
        ln_sums = logsumexp(M, axis=1, keepdims=True)
        out = M - ln_sums
        out[np.isneginf(ln_sums.squeeze())] = -np.inf
        return out

    @staticmethod
    @njit
    def count_transitions(T, n_states, states):
        A = np.full((n_states, n_states), -np.inf)
        for t in range(T - 1):
            i, j = states[t], states[t+1]
            A[i, j] = np.logaddexp(A[i, j], 0.0)
        return A
    
    @staticmethod
    @njit
    def count_emissions(T, n_states, vocab_size, states, seq):
        E = np.full((n_states, vocab_size), -np.inf)
        for t in range(T):
            s, k = states[t], seq[t]
            E[s, k] = np.logaddexp(E[s, k], 0.0)
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




