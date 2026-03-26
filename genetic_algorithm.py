import numpy as np
from scipy.spatial.distance import hamming
import Levenshtein
from pathlib import Path
import polars as pl
import polars.selectors as cs

home = Path(__file__).parent
data_path = home / "popns"

objectives = {
    "cos_dist": {"reverse": False},
    "ham_dist": {"reverse": False},
    "hmm_prob": {"reverse": False}
}


class NSGA_II:

    def __init__(self, pop_name: str, objectives: dict, hmm: object, loader: object, model=None, tokenizer=None, pop_cap=1000, idx=None):
        self.loader = loader
        self.model = model
        self.tokenizer = tokenizer
        self.hmm = hmm
        self.fitness_functions = {
            "cos_dist": lambda x: self.cos_dist(x),
            "ham_dist": lambda x: self.ham_dist(x),
            "hmm_prob": lambda x: self.hmm_prob(x),
            "levenshtein": lambda x: self.lev_dist(x),
            "conv_dev": lambda x: self.conv_dev(x),
            "conv_sum": lambda x: self.conv_sum(x),
        }
        if not set(objectives.keys()).issubset(self.fitness_functions.keys()):
                raise ValueError("Choose proper fitness functions.")
        self.objectives = objectives

        data_path.mkdir(exist_ok=True, parents=True)
        self.pop_name = pop_name
        self.popn_path = data_path / f"{self.pop_name}.parquet"

        self.popn = []
        self.pop_cap = pop_cap
        if idx != None:
            self.idx = idx
        else:
            self.idx = np.random.randint(1, len(self.loader))
        
        self.df = pl.DataFrame()
        self.state_choices = np.array(self.loader.states)
        self.k = self.hmm.k


    def gen_entry(self):
        data = self.loader.get_idx(self.idx, with_row=True)
        entry = {"annot_idx": self.idx,
                 "raw_idx": data["start"],
                 "orig_seq": list(data["seq"]),
                 "orig_states": data["states"],
                 "orig_conv": data["conv"],
                 "orig_embedding": self.get_embedding(data["seq"]),
                 "orig_ll": self.get_likelihood_only(data["seq"]),
                 "mod_seq": list(data["seq"]),
                 "mod_states": data["states"].copy(),
                 "mod_conv_score": data["conv"].copy(),
                 "mod_embedding": self.get_embedding if self.model != None else None,
                 "mut_array": np.zeros(len(data["seq"])),
                 "mod_ll": -np.inf,
                 "fitness": {key: 0 for key in self.objectives},
                 "live": True}
        return entry
    
    def get_likelihood_only(self, seq):
        enc_seq = self.hmm.tokenize(seq)
        p_x = self.hmm.ll_only(enc_seq)
        return p_x

    def get_embedding(self, seq):
        if self.model is not None and self.tokenizer is not None:
            inputs = self.tokenizer(seq, return_tensors = 'pt')["input_ids"]
            hidden_states = self.model(inputs)[0] # [1, sequence_length, 768]
            embedding_mean = hidden_states[0].detach().numpy().mean(axis=0)
            return embedding_mean
        return None

    def load_popn(self):
        if self.popn_path.exists():
            df = pl.read_parquet(self.popn_path)
            popn = df.filter(pl.col("live" == True))
            fit_df = popn.select(cs.startswith("fit_"))
            fit_df = fit_df.rename({col: col.removeprefix("fit_") for col in fit_df.columns})
            for row, fitness in zip(popn.iter_rows(named=True), fit_df.iter_rows(named=True)):
                row["fitness"] = fitness
                self.popn.append(row)
            self.df = df
        else:
            for i in range(self.pop_cap):
                self.popn.append(self.gen_entry())
    
    def save_popn(self, indices=None, write=False):
        indices = list(range(len(self.popn))) if indices is None else indices

        rows = []
        for i in indices:
            entry = self.popn[i]
            row = {k: v for k, v in entry.items() if k != "fitness"}
            for k, v in entry["fitness"].items():
                row[f"fit_{k}"] = v
            rows.append(row)

        current_popn = pl.DataFrame(rows)
        self.df = pl.concat([self.df, current_popn])
        if write:
            self.df.write_parquet(self.popn_path)
    
    def non_dominated_sort(self):
        fitness_arrays = [
            (i, np.array([
                -entry["fitness"][o] if self.objectives[o]["reversed"] else entry["fitness"][o] for o in self.objectives
            ]))
            for i, entry in enumerate(self.popn)
        ]

        



    
    def truncate_popn(self):
        if len(self.popn) > self.pop_cap:
            survivor_indices = self.non_dominated_sort()
            unalive = list(set(range(len(self.popn))) - set(survivor_indices))
            self.save_popn(unalive)
            for i in sorted(unalive, reverse=True):
                self.popn.pop(i)




        
    
    def cos_dist(self, entry):
        A = entry["orig_embedding"]
        B = entry["mod_embedding"]
        norm_a = A / np.linalg.norm(A)
        norm_b = B / np.linalg.norm(B)
        cdist = 1 - np.dot(norm_a, norm_b)
        return cdist
    
    def ham_dist(self, entry):
        return hamming (list(entry["orig_seq"]), list(entry["mod_seq"]))

    def lev_dist(self, entry):
        return Levenshtein.distance(entry["orig_seq"], entry["mod_seq"])
    
    def hmm_prob(self, entry):
        return entry["mod_ll"]
    
    def conv_dev(self, entry):
        total = entry["orig_conv"].sum()
        mut_score = entry["orig_conv"][entry["mod_array"] != 0]
        return mut_score / total
    
    def conv_sum(self, entry):
        return entry["orig_conv"][entry["mod_array"] != 0]

    def get_fitness(self, entry):
        for objective in self.objectives:
            entry["fitness"][objective] = self.fitness_functions[objective](entry)

    def mutate_prefix(self, mut_idx, idx):
        lmer = self.popn[idx]["mod_seq"][mut_idx - self.k + 1:mut_idx]
        kmer = self.popn[idx]["mod_seq"][mut_idx - self.k + 1:mut_idx + 1]
        nuc, state = self.hmm.prefixed_gen(lmer, kmer, self.popn[idx]["orig_states"][mut_idx], self.popn[idx]["orig_states"][mut_idx - 1])
        return nuc, state
    
    def update_arrays(self, nucleotide, state, mut_idx, idx):
        self.popn[idx]["mod_seq"][mut_idx] = nucleotide
        self.popn[idx]["mod_states"][mut_idx] = state
        self.popn[idx]["mut_array"][mut_idx] = 1
    
    def revise_next(self, mut_idx, idx):
        state = self.popn[idx]["mod_states"][mut_idx - 1]
        lmer = self.popn[idx]["mod_seq"][mut_idx - self.k + 2:mut_idx + 1]
        nuc = self.hmm.fixed_state_gen(lmer, state)
        self.update_arrays(nuc, state, mut_idx)

    def check_next(self, mut_idx, idx):
        mut_idx += 1
        if mut_idx < len(self.popn[idx]["mod_seq"]):
            state = self.popn[idx]["mod_states"][mut_idx]
            kmer = self.popn[idx]["mod_seq"][mut_idx - self.k + 1:mut_idx + 1]
            return self.hmm.check_next_nucleotide(kmer, state)
        return False

    def random_mutate_entry(self, idx, max_tries=100):
        
        def mutate(mut_idx, x):
            attempt = 0
            nuc, state = self.mutate_prefix(mut_idx, idx)
            while nuc is None and attempt < max_tries:
                if x - 1 > 0:
                    x -= 1
                elif x + 1 < len(indices[mask]):
                    x += 1
                else:
                    x = np.random.choice(np.where(mask)[0])
                mut_idx = indices[x]
                nuc, state = self.mutate_prefix(mut_idx, idx)
                attempt += 1
            self.update_arrays(nuc, state, mut_idx, idx)
            return mut_idx

        indices = np.where(self.popn[idx]["mut_array"] == 0)[0]
        mask = np.arange(len(indices)) >= (self.k - 1)
        choices = np.where(mask)[0]
        if len(choices) > 0:
            x = np.random.choice(choices)
            mut_idx = indices[x]
    
            mut_idx = mutate(mut_idx, x)
            while mut_idx + 1 < len(self.popn[idx]["mod_seq"]) and self.check_next(mut_idx, idx) == False:
                self.revise_next(mut_idx, idx)
                mut_idx += 1
        
        # self.get_fitness(self.popn[idx])
    
    def crossover(self, idx1, idx2, seg=2):

        def heal_boundaries(points, child_seq, child_state, child_mut):
            mut_idx = points[i]
            kmer = child_seq[mut_idx - self.k + 1:mut_idx + 1]
            lmer = child_seq[mut_idx - self.k + 1:mut_idx]
            state = child_state[mut_idx]
            while mut_idx < points[i + 1] and self.hmm.check_next_nucleotide(kmer, state) == False:
                prev_state = child_state[mut_idx - 1]
                nuc = self.hmm.fixed_state_gen(lmer, prev_state)
                child_state[mut_idx] = prev_state
                child_mut[mut_idx] = 1
                child_seq[mut_idx] = nuc
                mut_idx += 1
                kmer = child_seq[mut_idx - self.k + 1:mut_idx + 1]
                lmer = child_seq[mut_idx - self.k + 1:mut_idx]
                state = child_state[mut_idx]
        
        def make_child(n, o, m):
            child = self.gen_entry()
            child["mod_seq"] = child["orig_seq"][:self.k - 1] + n
            child["mod_states"] = o
            child["mut_array"] = m
            child["mod_ll"] = self.get_likelihood_only(child["mod_seq"])
            child["mod_embedding"] = self.get_embedding if self.model != None else None
            return child


        seq1 = self.popn[idx1]["mod_seq"].copy()
        seq2 = self.popn[idx2]["mod_seq"].copy()

        state1 = self.popn[idx1]["mod_states"].copy()
        state2 = self.popn[idx2]["mod_states"].copy()

        mut1 = self.popn[idx1]["mut_array"].copy()
        mut2 = self.popn[idx2]["mut_array"].copy()
        
        T = len(seq1)
        M = len(mut1)

        points = [self.k - 1] + sorted(np.random.choice(range(self.k - 1, T), seg, replace=False)) + [T]

        n1, n2 = [], []
        m1, m2 = np.zeros(M, dtype=np.uint8), np.zeros(M, dtype=np.uint8)
        o1, o2 = np.zeros(M, dtype=np.uint8), np.zeros(M, dtype=np.uint8)
        for i in range(len(points) - 1):
            if i % 2 == 0:
                s1, s2 = seq1, seq2
                q1, q2 = mut1, mut2
                p1, p2 = state1, state2
            else:
                s1, s2 = seq2, seq1
                q1, q2 = mut2, mut1
                p1, p2 = state2, state1
            
            n1.extend(s1[points[i]: points[i+1]])
            n2.extend(s2[points[i]: points[i+1]])
            m1[points[i]:points[i+1]] = q1[points[i]:points[i+1]]
            m2[points[i]:points[i+1]] = q2[points[i]:points[i+1]]
            o1[points[i]:points[i+1]] = p1[points[i]:points[i+1]]
            o2[points[i]:points[i+1]] = p2[points[i]:points[i+1]]

            if i > 0:
                heal_boundaries(points, n1, o1, m1)
                heal_boundaries(points, n2, o2, m2)

        self.popn.append(make_child(n1, o1, m1))
        self.popn.append(make_child(n2, o2, m2))










        
            

        




        
        
        

    


