import numpy as np
from scipy.spatial.distance import hamming
import Levenshtein
from pathlib import Path
import polars as pl
import polars.selectors as cs
from numba import njit
from numba.typed import List as JitList
from numba import int32, typeof, types
from numba import njit, config
config.BOUNDSCHECK = True
import torch
import math
import time
import sys

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

    def __init__(self, pop_name: str, objectives: dict, hmm: object, loader: object, model=None, tokenizer=None, pop_cap=1000, idx=None, use_jacobian=False, batch_size=100):
        self.loader = loader
        self.model = model
        self.tokenizer = tokenizer
        self.jacobian = use_jacobian
        self.batch_size = batch_size

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
        entry =  {"annot_idx": self.idx,
                 "raw_idx": data["start"],
                 "seq": list(data["seq"]),
                 "states": data["states"],
                 "conv_array": data["conv"],
                 "ll": self.get_likelihood_only(data["seq"], data["states"][self.k - 2])}
        if self.cuda:
            embedding, sensitivities, token_lengths = self._encode_cuda([data["seq"]])
            if self.jacobian:
                sen_arr = np.array(sensitivities[0])
                entry["sensitivity"] = sen_arr
                entry["sen_weights"] = self._compute_sen_weights(sen_arr, token_lengths[0])
        else:
            embedding = self._encode_cpu([data["seq"]])
        entry["embedding"] = embedding[0]
        return entry
        
        

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

    # DNABert version
    # def _encode_cuda(self, seqs):
    #     if self.model is not None and self.tokenizer is not None:
    #         device = next(self.model.parameters()).device  # wherever the model lives
    #         inputs = self.tokenizer(seqs, return_tensors='pt', padding=True, truncation=False)["input_ids"]
    #         inputs = inputs.to(device)  # move input to same device as model
    #         with torch.no_grad():
    #             hidden_states = self.model(inputs)[0]
    #         embedding_mean = hidden_states.detach().cpu().numpy().mean(axis=1)  # mean over sequence dim
    #         return embedding_mean
    #     return [None for _ in range(len(seqs))]

    # def _encode_cuda(self, seqs, batch_size=24):
    #     if self.model is not None and self.tokenizer is not None:
    #         device = next(self.model.parameters()).device
    #         all_embeddings = []
    #         for i in range(0, len(seqs), batch_size):
    #             batch = seqs[i:i + batch_size]
    #             inputs = self.tokenizer(batch, return_tensors='pt', padding=True, truncation=False)
    #             input_ids = inputs["input_ids"].to(device)
    #             attention_mask = inputs["attention_mask"].to(device)
    #             with torch.no_grad():
    #                 hidden_states = self.model(input_ids, attention_mask=attention_mask, output_hidden_states=True).hidden_states[-1]
    #             attention_mask = attention_mask.unsqueeze(-1)
    #             emb = (attention_mask * hidden_states).sum(axis=1) / attention_mask.sum(axis=1)
    #             all_embeddings.append(emb.detach().cpu().numpy())
    #             torch.cuda.empty_cache()
    #         return np.concatenate(all_embeddings, axis=0)
    #     return [None for _ in range(len(seqs))]

    def _compute_sen_weights(self, sen_arr, token_lengths):
        token_lengths = token_lengths[1:]
        sen_arr = sen_arr[1:]

        if sen_arr.max() == sen_arr.min():
            total = sum(token_lengths)
            return np.ones(total) / total
        sen_norm = (sen_arr - sen_arr.min()) / (sen_arr.max() - sen_arr.min())
        inv_sen = 1 - sen_norm
        weights = inv_sen / inv_sen.sum()
        return np.repeat(weights, token_lengths)

    def _encode_cuda(self, seqs):

        if self.model is not None and self.tokenizer is not None:
            device = next(self.model.parameters()).device
            max_length = self.tokenizer.model_max_length
            
            tokenized = [(i, s, self.tokenizer(s)["input_ids"]) for i, s in enumerate(seqs)]
            short_seqs = [(i, s) for i, s, ids in tokenized if len(ids) <= max_length]
            long_seqs = [(i, s) for i, s, ids in tokenized if len(ids) > max_length]
            
            all_embeddings = [None] * len(seqs)
            all_jacobians = [None] * len(seqs)
            
            # batch short sequences
            for i in range(0, len(short_seqs), self.batch_size):
                batch_indices, batch_seqs = zip(*short_seqs[i:i + self.batch_size])
                inputs = self.tokenizer(list(batch_seqs), return_tensors='pt', padding=True)
                input_ids = inputs["input_ids"].to(device)
                mask = inputs["attention_mask"].to(device)

                if self.jacobian:
                    
                    hidden_states = self.model(input_ids, attention_mask=mask, output_hidden_states=True).hidden_states[-1]

                    mask = mask.unsqueeze(-1)
                    embs = (mask * hidden_states).sum(axis=1) / mask.sum(axis=1)
                    embs.sum().backward()

                    sensitivities = self.model.esm.embeddings.word_embeddings.weight.grad[input_ids].norm(dim=-1).detach().cpu().clone()
                    embs = embs.detach().cpu().numpy()

                    del hidden_states
                    torch.cuda.empty_cache()
                    self.model.zero_grad()

                    for j, idx in enumerate(batch_indices):
                        all_embeddings[idx] = embs[j]
                        all_jacobians[idx] = sensitivities[j].detach().cpu().numpy()
                else:
                    with torch.no_grad():
                        hidden_states = self.model(input_ids, attention_mask=mask, output_hidden_states=True).hidden_states[-1]
                    mask = mask.unsqueeze(-1)
                    embs = (mask * hidden_states).sum(axis=1) / mask.sum(axis=1)

                    for j, idx in enumerate(batch_indices):
                        all_embeddings[idx] = embs[j].detach().cpu().numpy()
            
            # process long sequences individually with chunking
            for idx, seq in long_seqs:
                input_ids = self.tokenizer(seq, return_tensors='pt')["input_ids"][0]
                chunks = [input_ids[r:r + max_length] for r in range(0, len(input_ids), max_length)]
                chunk_ids = torch.nn.utils.rnn.pad_sequence(chunks, batch_first=True).to(device)
                mask = (chunk_ids != self.tokenizer.pad_token_id).long().to(device)

                if self.jacobian:
                    input_grads = []
                    def save_input_grad(module, grad_input, grad_output):
                        input_grads.append(grad_output[0].detach())

                    handle = self.model.esm.embeddings.word_embeddings.register_backward_hook(save_input_grad)

                    hidden_states = self.model(chunk_ids, attention_mask=mask, output_hidden_states=True).hidden_states[-1]
                    mask = mask.unsqueeze(-1)
                    chunk_embs = (mask * hidden_states).sum(axis=1) / mask.sum(axis=1)
                    chunk_embs.sum().backward()

                    handle.remove()

                    sensitivities = input_grads[0].norm(dim=-1).detach().cpu().clone()
                    chunk_embs = chunk_embs.detach().cpu().numpy()

                    del hidden_states
                    torch.cuda.empty_cache()
                    self.model.zero_grad()

                    all_embeddings[idx] = chunk_embs.mean(axis=0)
                    all_jacobians[idx] = torch.cat([sensitivities[k][:len(chunks[k])] for k in range(len(chunks))]).detach().cpu().numpy()
                else:
                    with torch.no_grad():
                        hidden_states = self.model(chunk_ids, attention_mask=mask, output_hidden_states=True).hidden_states[-1]
                    mask = mask.unsqueeze(-1)
                    chunk_embs = (mask * hidden_states).sum(axis=1) / mask.sum(axis=1)
                    all_embeddings[idx] = chunk_embs.mean(axis=0).detach().cpu().numpy()
            
            if self.jacobian:
                token_lengths = []
                for i, s, ids in tokenized:
                    tokens = self.tokenizer.convert_ids_to_tokens(ids)
                    token_lengths.append([len(t) for t in tokens])
            else:
                token_lengths = None

            torch.cuda.empty_cache()
            return np.array(all_embeddings), all_jacobians, token_lengths
        return [None for _ in range(len(seqs))], None, None

    def _encode_cpu(self, seqs):
        if self.model is not None and self.tokenizer is not None:
            embeddings = []
            for seq in seqs:
                inputs = self.tokenizer(seq, return_tensors='pt')["input_ids"]
                with torch.no_grad():
                    hidden_states = self.model(inputs)[0]
                embeddings.append(hidden_states[0].detach().numpy().mean(axis=0))
            return np.array(embeddings)
        return [None for _ in range(len(seqs))]

    def get_embeddings(self):
        indices = [i for i in range(len(self.popn)) if self.popn[i]["live"] and self.popn[i]["update"]]
        seqs = ["".join(self.popn[i]["seq"]) for i in indices]
        if self.cuda:
            embeddings, sensitivities, token_lengths = self._encode_cuda(seqs)
        else:
            embeddings = self._encode_cpu(seqs)
        if embeddings is not None:
            for i, idx in enumerate(indices):
                self.popn[idx]["embedding"] = embeddings[i]
                if self.jacobian:
                    sen_arr = np.array(sensitivities[i])
                    self.popn[idx]["sensitivity"] = sen_arr
                    self.popn[idx]["sen_weights"] = self._compute_sen_weights(sen_arr, token_lengths[i])
    
    def get_mod_ll(self):
        indices = [i for i in range(len(self.popn)) if self.popn[i]["live"] and self.popn[i]["update"]]
        lls = [self.get_likelihood_only(self.popn[i]["seq"], self.master["states"][self.k - 2]) for i in indices]
        for i, idx in enumerate(indices):
            self.popn[idx]["ll"] = lls[i]
    
    # def get_mod_conv_score(self):
    #     indices = [i for i in range(len(self.popn)) if self.popn[i]["live"] and self.popn[i]["update"]]
    #     masks = [np.where(self.popn[i]["mut_array"] > 0)[0] for i in indices]
    #     scores = [self.master["conv_array"][masks[i]].sum() / self.master["conv_array"].sum() for i, idx in enumerate(indices)]
    #     for i, idx in enumerate(indices):
    #         self.popn[idx]["conv_score"] = scores[i]

    # def get_mod_conv_score(self):
    #     indices = [i for i in range(len(self.popn)) if self.popn[i]["live"] and self.popn[i]["update"]]
    #     total = self.master["conv_array"].sum()
    #     for idx in indices:
    #         self.popn[idx]["conv_score"] = (self.master["conv_array"] * (self.popn[idx]["mut_array"] > 0)).sum() / total

    def get_mod_conv_score(self):
        indices = [i for i in range(len(self.popn)) if self.popn[i]["live"] and self.popn[i]["update"]]
        total = self.master["conv_array"].sum()
        mut_matrix = np.vstack([self.popn[i]["mut_array"] for i in indices])
        scores = mut_matrix.dot(self.master["conv_array"]) / total

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
    @njit(cache=True)
    def _fast_sort(fit_matrix, pop_cap, inner_template):
        N = fit_matrix.shape[0]

        #compute dominance relationships and first front
        fronts = JitList.empty_list(types.ListType(inner_template))
        first_front = JitList.empty_list(int32)
        fronts.append(first_front)

        # fronts = [[]]  # regular Python lists
        # first_front = []
        # fronts[0] = first_front

        dom_counts = np.zeros(N, dtype=np.int32)
        dom_matrix = np.zeros((N, N), dtype=np.uint8)
        for i in range(N):
            p = fit_matrix[i, :]
            for j in range(i + 1, N):
                q = fit_matrix[j, :]
                if np.all(p >= q) and np.any(p > q): # non-strict dominance
                    dom_matrix[i][j] = 1
                    dom_counts[j] += 1
                elif np.all(q >= p) and np.any(q > p):
                    dom_matrix[j][i] = 1
                    dom_counts[i] += 1
            if dom_counts[i] == 0:
                fronts[0].append(np.int32(i))
        
        #find next front
        i = 0
        while len(fronts[i]) > 0:
            front = JitList.empty_list(int32)
            # front = []
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

        last_front = np.zeros(0, dtype=np.int32)
        # last_front = []
        # print(f"len(fronts[0])={len(fronts[0])}, pop_cap={pop_cap}")
        if len(fronts[0]) > pop_cap:

            result = np.zeros(len(fronts[0]), dtype=np.int32)
            for i in range(len(fronts[0])):
                result[i] = fronts[0][i]
            
            return last_front, pop_cap, result

        #concat individuals and 
        sorted_individuals = np.zeros(N, dtype=np.int32)
        count = 0
        for i in range(len(fronts)):
            if count + len(fronts[i]) <= pop_cap:
                for j in range(len(fronts[i])):
                    if count >= N:
                        break
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
                    last_front = np.zeros(len(fronts[i]), dtype=np.int32)
                    for k in range(len(fronts[i])):
                        last_front[k] = fronts[i][k]
                    return sorted_individuals[:count], n_slots, last_front
        return sorted_individuals[:count], 0, last_front
    
    @staticmethod
    @njit(cache=True)
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
            fit_matrix_sliced = fit_matrix[last_front, :] # need to normalize the fit_matrix for crowding distance
            mins = np.min(fit_matrix_sliced, axis=0)
            maxes = np.max(fit_matrix_sliced, axis=0)
            ranges = maxes - mins
            ranges[ranges == 0] = 1.0 # squeeze all values into 0-1 taking into account whether they are reversed or not
            fit_matrix_normalized = (fit_matrix_sliced - mins) / ranges

            last_front = self._crowding_distance(fit_matrix_normalized, n_slots, last_front)
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
        if len(indices) == 0:
            return
        mask = np.arange(len(indices)) >= (self.k - 1)
        choices = np.where(mask)[0]
        if len(choices) > 0:
            if self.jacobian:
                if "sen_weights" in self.popn[idx]:
                    masked_weights = self.popn[idx]["sen_weights"][indices]
                else:
                    masked_weights = self.master["sen_weights"][indices]
                masked_weights = masked_weights[mask]
                masked_weights = masked_weights / masked_weights.sum()
                if np.isnan(masked_weights).any():
                    # all selected positions are max sensitivity, fall back to uniform
                    masked_weights = np.ones(len(choices)) / len(choices)
                x = np.random.choice(choices, p=masked_weights)
            else:
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
        if not mask.any():
            return
        for i in range(seg):
            available = choice_array[mask]
            if len(available) == 0:
                break
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
            # seg = np.random.randint(2, 5)
            seg = 2
            self.crossover(pair[0], pair[1], seg)

            for index in (-1, -2):
                current_muts = self.popn[index]["mut_array"].sum()
                while self.popn[index]["mut_array"].sum() - current_muts < n_muts and (self.popn[index]["mut_array"][self.k - 1:] == 0).any():
                    self.random_mutate_entry(index)

    def update_entries(self):
        self.get_mod_ll()
        if "conv_dev" or "conv_sum" in self.objectives:
            self.get_mod_conv_score()
        self.get_embeddings()
        self.compute_total_fitness()
        self.set_update_flag()


        










        
            

        




        
        
        

    


