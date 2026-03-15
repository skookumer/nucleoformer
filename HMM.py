import numpy as np
from scipy.special import logsumexp
from pathlib import Path
from itertools import product
import random

home = Path(__file__).parent


class Encoder_Decoder:

    def __init__(self, sequence_map=['A', 'C', 'G', 'T', 'N', '-']):
        self.sequence_map = sequence_map
        self.base_map = self.init_base_encoding_map()
        self.decode_map = np.array(sequence_map)
    
    def encode_seq(self, seq, k=0):
        bytes = np.frombuffer("".join(seq).encode(), dtype=np.uint8)
        if k == 0:
            return np.array([self.base_map[bytes[i]] for i in range(len(bytes))])
        else:
            pad = [5 for _ in range(k - 1)]
            lmerized = pad + [self.base_map[bytes[i]] for i in range(len(bytes))] + pad
            return np.array(lmerized)

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
        self.vocab = self.dictionize(n_bases, k)
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
        return {v: k for k, v in enumerate(product(range(n_bases), repeat=k))}
    
    def tokenize(self, seq):
        win_size = self.k
        tokens = [self.vocab[tuple(seq[i:i+win_size])] for i in range(len(seq) - self.k + 1)]
        return np.array(tokens)


    def load_matrices(self):
        data = np.load(self.matrix_path)
        self.transition_matrix = data["transition_matrix"]
        self.emission_matrix = data["emission_matrix"]
        self.pi = data["pi"]
         
    def init_matrices(self):   
        N = self.n_states    
        self.transition_matrix = self.log_norm_random((N, N))
        self.emission_matrix = self.log_norm_random((N, self.vocab_size))
        self.pi = self.log_norm_random((N,))
    
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

        V = np.full((N, T) - np.inf)
        BP = np.zeros((N, T), dtype=int)

        V[:, 0] = self.pi + self.emission_matrix[:, seq[0]]     #initial probs + first observation

        for t in range(1, T):
            for s in range(N):
                scores = V[:, t - 1] + self.transition_matrix[:, s]
                best_score = np.argmax(scores)
                V[s, t] = scores[best_score] + self.emission_matrix[s, seq[t]]
                BP[s, t] = best_score
        
        best_path = np.zeros(T, dtype=int)
        best_path[T-1] = np.argmax(V[: T-1])
        
        for t in range(T-2, -1, -1):
            best_path[t] = BP[best_path[t+1], t+1]
        
        return best_path

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

        for i in range(iterations):
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

    def train_model(self, seqs, epochs=3, toprint=False):

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
    print(model.emission_matrix)
    print(model.transition_matrix)
    print(model.pi)
    # model.viterbi()




