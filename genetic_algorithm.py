import numpy as np
from scipy.spatial.distance import hamming
import Levenshtein
from pathlib import Path
import polars as pl
import polars.selectors as cs
from numba import njit
from numba.typed import List as JitList
from numba import int32, typeof, types
import torch
import math

home = Path(__file__).parent
data_path = home / "popns"

objectives = {
    "cos_dist": {"reverse": False},
    "ham_dist": {"reverse": False},
    "hmm_prob": {"reverse": False}
}


'''
look into BLOSUM scores for substitution severity; scale phylop scores by this?
'''


class NSGA_II:

    def __init__(self, pop_name: str, objectives: dict, hmm: object, loader: object, model=None, tokenizer=None, pop_cap=1000, idx=None):
        self.loader = loader
        self.model = model
        self.tokenizer = tokenizer

        if self.model is not None and self.tokenizer is not None:
            if next(self.model.parameters()).is_cuda:
                self.cuda = True
            else:
                self.cuda = False
        else:
            self.cuda = False
        
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
        self.master = self.make_master_dict()



    def make_master_dict(self):
        data = self.loader.get_idx(self.idx, with_row=True)
        return {"annot_idx": self.idx,
                 "raw_idx": data["start"],
                 "seq": list(data["seq"]),
                 "states": data["states"],
                 "conv_array": data["conv"],
                 "embedding": self._encode_cuda([data["seq"]])[0] if self.cuda else self._encode_cpu([data["seq"]])[0],
                 "ll": self.get_likelihood_only(data["seq"], data["states"][self.k - 2])}
        

    def gen_entry(self):
        entry = {"seq": self.master["seq"].copy(),
                 "states": self.master["states"].copy(),
                 "conv_score": 0,
                 "embedding": None,
                 "mut_array": np.zeros(len(self.master["seq"]), dtype=np.uint8),
                 "ll": -np.inf,
                 "fitness": {key: 0 for key in self.objectives},
                 "live": True,
                 "update": True}
        return entry
    
    def get_likelihood_only(self, seq, initial_state):
        enc_seq = self.hmm.tokenize(seq)
        p_x = self.hmm.ll_only(enc_seq, initial_state)
        return p_x

    def _encode_cuda(self, seqs):
        if self.model is not None and self.tokenizer is not None:
            device = next(self.model.parameters()).device  # wherever the model lives
            inputs = self.tokenizer(seqs, return_tensors='pt', padding=True, truncation=True)["input_ids"]
            inputs = inputs.to(device)  # move input to same device as model
            with torch.no_grad():
                hidden_states = self.model(inputs)[0]
            embedding_mean = hidden_states.detach().cpu().numpy().mean(axis=1)  # mean over sequence dim
            return embedding_mean
        return [None for _ in range(len(seqs))]

    def _encode_cpu(self, seqs):
        if self.model is not None and self.tokenizer is not None:
            embeddings = []
            for seq in seqs:
                inputs = self.tokenizer(seq, return_tensors='pt')["input_ids"]
                hidden_states = self.model(inputs)[0]
                embeddings.append(hidden_states[0].detach().numpy().mean(axis=0))
            return np.array(embeddings)
        return [None for _ in range(len(seqs))]

    def get_embeddings(self):
        indices = [i for i in range(len(self.popn)) if self.popn[i]["live"] and self.popn[i]["update"]]
        seqs = ["".join(self.popn[i]["seq"]) for i in indices]
        if self.cuda:
            embeddings = self._encode_cuda(seqs)
        else:
            embeddings = self._encode_cpu(seqs)
        if embeddings is not None:
            for i, idx in enumerate(indices):
                self.popn[idx]["embedding"] = embeddings[i]
    
    def get_mod_ll(self):
        indices = [i for i in range(len(self.popn)) if self.popn[i]["live"] and self.popn[i]["update"]]
        lls = [self.get_likelihood_only(self.popn[i]["seq"], self.master["states"][self.k - 2]) for i in indices]
        for i, idx in enumerate(indices):
            self.popn[idx]["ll"] = lls[i]
    
    def get_mod_conv_score(self):
        indices = [i for i in range(len(self.popn)) if self.popn[i]["live"] and self.popn[i]["update"]]
        masks = [np.where(self.popn[i]["mut_array"] > 0)[0] for i in indices]
        scores = [self.master["conv_array"][masks[i]].sum() / self.master["conv_array"].sum() for i, idx in enumerate(indices)]
        for i, idx in enumerate(indices):
            self.popn[idx]["conv_score"] = scores[i]
    
    def set_update_flag(self):
        for entry in self.popn:
            entry["update"] = False

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
        
    @staticmethod
    @njit(cache=False)
    def _fast_sort(fit_matrix, pop_cap, inner_template):
        N = fit_matrix.shape[0]

        #compute dominance relationships and first front
        fronts = JitList.empty_list(types.ListType(inner_template))
        first_front = JitList.empty_list(int32)
        fronts.append(first_front)

        dom_counts = np.zeros(N, dtype=np.int32)
        dom_matrix = np.zeros((N, N), dtype=np.uint8)
        for i in range(N):
            p = fit_matrix[i, :]
            for j in range(i + 1, N):
                q = fit_matrix[j, :]
                if np.all(p > q):
                    dom_matrix[i][j] = 1
                    dom_counts[j] += 1
                elif np.all(q > p):
                    dom_matrix[j][i] = 1
                    dom_counts[i] += 1
            if dom_counts[i] == 0:
                fronts[0].append(np.int32(i))
        
        #find next front
        i = 0
        while len(fronts[i]) > 0:
            front = JitList.empty_list(int32)
            for j in range(len(fronts[i])):
                p = fronts[i][j]
                p_dominates_q = np.where(dom_matrix[p, :] > 0)[0]
                for k in range(len(p_dominates_q)):
                    q = p_dominates_q[k]
                    dom_counts[q] -= 1
                    if dom_counts[q] == 0:
                        front.append(np.int32(q))
            i += 1
            fronts.append(front)

        last_front = JitList.empty_list(int32)
        if len(fronts[0]) >= pop_cap:

            result = np.zeros(pop_cap, dtype=np.int32)
            for i in range(len(fronts[0])):
                result[i] = fronts[0][i]

            return result, 0, last_front

        #concat individuals and 
        sorted_individuals = np.zeros(N, dtype=np.int32)
        count = 0
        for i in range(len(fronts)):
            if count + len(fronts[i]) <= pop_cap:
                for j in range(len(fronts[i])):
                    sorted_individuals[count] = fronts[i][j]
                    count += 1
            else:
                n_slots = pop_cap - count
                if len(fronts[i]) <= 2:
                    for k in range(min(n_slots, len(fronts[i]))):
                        sorted_individuals[count] = fronts[i][k]
                        count += 1
                    return sorted_individuals[:count], 0, last_front
                else:
                    return sorted_individuals[:count], n_slots, fronts[i]
        return sorted_individuals[:count], 0, last_front
    
    @staticmethod
    @njit(cache=False)
    def _crowding_distance(fit_matrix, n_slots, last_front_list):
        last_front = np.zeros(len(last_front_list), dtype=np.int32)
        for i in range(len(last_front_list)):
            last_front[i] = last_front_list[i]
        T = fit_matrix.shape[0]
        C = fit_matrix.shape[1]
        dists = np.zeros(T, dtype=np.float64)
        for c in range(C):
            col = fit_matrix[:, c]
            sorted_row_indices = np.argsort(col)

            highest = sorted_row_indices[-1]
            lowest = sorted_row_indices[0]
            max_range = fit_matrix[highest, c] - fit_matrix[lowest, c]
            if max_range != 0:
                dists[highest] = np.inf
                dists[lowest] = np.inf
                for i in range(1, T - 1):
                    p = sorted_row_indices[i - 1]
                    q = sorted_row_indices[i + 1]
                    r = sorted_row_indices[i]
                    if dists[r] != np.inf:
                        dists[r] += (fit_matrix[q, c] - fit_matrix[p, c]) / max_range
        sorted_indices = np.argsort(dists)[::-1][:n_slots]
        return last_front[sorted_indices]
    
    def non_dominated_sort(self):
                
        fit_matrix = np.array([np.array([
            -entry["fitness"][o] if self.objectives[o]["reverse"] else entry["fitness"][o] for o in self.objectives
            ]) for entry in self.popn])
        
        sorted_individuals, n_slots, last_front = self._fast_sort(fit_matrix, self.pop_cap, JitList.empty_list(int32))
        if n_slots > 0:
            last_front = self._crowding_distance(fit_matrix[last_front, :], n_slots, last_front)
            sorted_individuals = np.concatenate([sorted_individuals, last_front])
        return sorted_individuals #an array of sorted individual indices
        

    def truncate_popn(self):
        if len(self.popn) > self.pop_cap:
            survivor_indices = self.non_dominated_sort()
            unalive = list(set(range(len(self.popn))) - set(survivor_indices))
            for idx in unalive:
                self.popn[idx]["live"] = False
            for i in sorted(unalive, reverse=True):
                self.popn.pop(i)
            

    def cos_dist(self, entry):
        A = self.master["embedding"]
        B = entry["embedding"]
        norm_a = A / np.linalg.norm(A)
        norm_b = B / np.linalg.norm(B)
        cdist = 1 - np.dot(norm_a, norm_b)
        return cdist
    
    def ham_dist(self, entry):
        return hamming (list(self.master["seq"]), list(entry["seq"]))

    def lev_dist(self, entry):
        return Levenshtein.distance(self.master["seq"], entry["seq"])
    
    def hmm_prob(self, entry):
        return entry["ll"] - self.master["ll"]
    
    def conv_dev(self, entry):
        total = self.master["conv_array"].sum()
        mut_score = self.master["conv_array"][entry["mut_array"] != 0]
        return mut_score / total
    
    def conv_sum(self, entry):
        return self.master["conv_array"][entry["mut_array"] != 0]

    def get_fitness(self, entry):
        for objective in self.objectives:
            entry["fitness"][objective] = self.fitness_functions[objective](entry)
    
    def compute_total_fitness(self):
        for i in range(len(self.popn)):
            if self.popn[i]["live"] == True and self.popn[i]["update"] == True:
                self.get_fitness(self.popn[i])

    def mutate_prefix(self, mut_idx, idx):
        lmer = self.popn[idx]["seq"][mut_idx - self.k + 1:mut_idx]
        kmer = self.popn[idx]["seq"][mut_idx - self.k + 1:mut_idx + 1]
        nuc, state = self.hmm.prefixed_gen(lmer, kmer, self.master["states"][mut_idx], self.master["states"][mut_idx - 1])
        return nuc, state
    
    def update_arrays(self, nucleotide, state, mut_idx, idx):
        self.popn[idx]["seq"][mut_idx] = nucleotide
        self.popn[idx]["states"][mut_idx] = state
        self.popn[idx]["mut_array"][mut_idx] = 1
    
    def revise_next(self, mut_idx, idx):
        state = self.popn[idx]["states"][mut_idx - 1]
        lmer = self.popn[idx]["seq"][mut_idx - self.k + 2:mut_idx + 1]
        nuc = self.hmm.fixed_state_gen(lmer, state)
        self.update_arrays(nuc, state, mut_idx)

    def check_next(self, mut_idx, idx):
        mut_idx += 1
        if mut_idx < len(self.popn[idx]["seq"]):
            state = self.popn[idx]["states"][mut_idx]
            kmer = self.popn[idx]["seq"][mut_idx - self.k + 1:mut_idx + 1]
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
            while mut_idx + 1 < len(self.popn[idx]["seq"]) and self.check_next(mut_idx, idx) == False:
                self.revise_next(mut_idx, idx)
                mut_idx += 1
            self.popn[idx]["update"] = True
        
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
            child["seq"] = self.master["seq"][:self.k - 1] + n
            child["states"] = o
            child["mut_array"] = m
            child["update"] = True
            return child


        seq1 = self.popn[idx1]["seq"].copy()
        seq2 = self.popn[idx2]["seq"].copy()

        state1 = self.popn[idx1]["states"].copy()
        state2 = self.popn[idx2]["states"].copy()

        mut1 = self.popn[idx1]["mut_array"].copy()
        mut2 = self.popn[idx2]["mut_array"].copy()
        
        T = len(seq1)
        M = len(mut1)

        choice_array = np.arange(T)
        mask = np.full(len(choice_array), True, dtype=bool)
        mask[0: self.k] = False
        mask[T - self.k: T] = False
        choices = []
        for i in range(seg):
            idx = np.random.choice(choice_array[mask])
            choices.append(idx)
            mask[max(0, idx - self.k): min(T, idx + self.k + 1)] = False

        points = [self.k - 1] + sorted(choices) + [T]

        n1, n2 = [], []
        m1, m2 = np.zeros(M, dtype=np.uint8), np.zeros(M, dtype=np.uint8)
        state_prefix = self.master["states"][:self.k - 1]
        o1 = np.concatenate([state_prefix, np.zeros(M - self.k + 1, dtype=np.uint8)])
        o2 = np.concatenate([state_prefix, np.zeros(M - self.k + 1, dtype=np.uint8)])
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

    def cross_mutate_random(self, cross_rate, mut_rate):
        indices = np.random.permutation(len(self.popn))
        if len(indices) % 2 != 0:
            indices = indices[:-1]
        pairs = indices.reshape(-1, 2)

        n_keep = round(cross_rate * len(pairs))
        pairs = pairs[:n_keep]
        
        if type(mut_rate) == float:
            n_muts = math.ceil(len(self.master["seq"]) * mut_rate)
        elif type(mut_rate) == int:
            n_muts = mut_rate

        for pair in pairs:
            seg = np.random.randint(2, 5)
            self.crossover(pair[0], pair[1], seg)

            for index in (-1, -2):
                current_muts = self.popn[index]["mut_array"].sum()
                while self.popn[index]["mut_array"].sum() - current_muts < n_muts and (self.popn[index]["mut_array"] == 0).any():
                    self.random_mutate_entry(index)

    def update_entries(self):
        self.get_mod_ll()
        self.get_mod_conv_score()
        self.get_embeddings()
        self.compute_total_fitness()
        self.set_update_flag()


        










        
            

        




        
        
        

    


