import sys
import polars as pl
import json
from pathlib import Path
from pyfaidx import Fasta
import pysam
import requests
from pprint import pprint
import numpy as np
from numba import njit
import pyBigWig

'''
Notes 3/24
enhancer above regulatory
ignore immune

parse regulatory to insulators, silencers, enhancers separately; list as regulatory element

intergenic can't be gene
region refers to entire chromosome
a primary transcript includes introns

Gene is a CDS that is comprised of exons and introns


Use the image
insulator, silencer, enhancher: regulatory regions of interest
'''

state_dict_old = {
    "promoter": ["promoter", "TATA_box", "CAAT_signal", "GC_rich_promoter_region", "CAGE_cluster", "nucleotide_motif", 
                 "nucleotide_cleavage_site", "conserved_region", "replication_start_site",],
    "immune": ["V_gene_segment", "D_gene_segment", "J_gene_segment", "C_gene_segment"],
    "regulatory": ["insulator", "silencer", "locus_control_region", "imprinting_control_region",
        "enhancer_blocking_element", "regulatory_region", "transcriptional_cis_regulatory_region",
        "response_element", "protein_binding_site", "matrix_attachment_site",
        "DNaseI_hypersensitive_site", "epigenetically_modified_region", "antisense_RNA"],
    "enhancer": ["enhancer"],
    "repeat": ["microsatellite", "minisatellite", "mobile_genetic_element", "tandem_repeat", "repeat_region",
               "dispersed_repeat", "direct_repeat", "repeat_instability_region", "centromere", "match"],
    "exon": ["exon", "CDS"],
    "ncRNA": ["lnc_RNA", "miRNA", "tRNA", "rRNA", "snRNA", "snoRNA", "scRNA", "scaRNA", 
            "Y_RNA", "RNase_MRP_RNA", "RNase_P_RNA", "telomerase_RNA", "vault_RNA", "pseudogene",],
    "intron": ["mRNA", 
            "primary_transcript", 
            "transcript",],
    "intergenic": ["gene", "region", "biological_region", "cDNA_match", "sequence_comparison",
             "sequence_feature", "sequence_alteration", "sequence_alteration_artifact",
             "chromosome_breakpoint", #"nucleotide_cleavage_site", "nucleotide_motif",
             "sequence_secondary_structure", "D_loop", #"conserved_region",
             "origin_of_replication", "replication_regulatory_region",#"replication_start_site", 
             "meiotic_recombination_region", "mitotic_recombination_region",
             "non_allelic_homologous_recombination_region", "recombination_feature",
            #  "microsatellite", "minisatellite", "tandem_repeat", "repeat_region",   #repeat
            #  "mobile_genetic_element", "dispersed_repeat", "direct_repeat", "repeat_instability_region", #repeat
            #  "mRNA", "primary_transcript", "transcript", #from exon
            #  "pseudogene", #from exon
            #  "V_gene_segment", "D_gene_segment", "J_gene_segment", "C_gene_segment" #immune
             ]
}

state_dict = {
    "CAGE_cluster": ["CAGE_cluster"],
    "promoter": ["synthetic_promoter", "nucleotide_motif", "nucleotide_cleavage_site", 
                 "DNaseI_hypersensitive_site", "protein_binding_site", "response_element", 
                "transcriptional_cis_regulatory_region", "GC_rich_promoter_region",
                "CpG_island", "CAAT_signal", "TATA_box"],
    "enhancer": ["enhancer"],
    "regulatory": ["silencer", "insulator"],
    "ncRNA": ["miRNA", "tRNA", "rRNA", "snRNA", "snoRNA", "scRNA", "scaRNA", 
            "Y_RNA", "RNase_MRP_RNA", "RNase_P_RNA", "telomerase_RNA", "vault_RNA", "pseudogene",],
    "repeat": ["microsatellite", "minisatellite", "mobile_genetic_element", "tandem_repeat", "repeat_region",
               "dispersed_repeat", "direct_repeat", "repeat_instability_region", "centromere"],
    "exon": ["exon", "CDS"],
    "lnc_RNA": ["lnc_RNA"],
    "intron": ["primary_transcript", "transcript", "mRNA"],
    "intergenic": ["gene", "region"]
}

if sys.platform == "win32":
    reference_path = Path("C:/Users/Eric Arnold/Documents/reference_genome")
else:
    reference_path = Path("/mnt/c/Users/Eric Arnold/Documents/reference_genome")

ncbi_path = reference_path / "ncbi_dataset/data/GCF_000001405.40"

reference_files = [f for f in reference_path.iterdir()]
names = {"hg19": "GCF_000001405.21_GRCh37.p9_genomic.fna"}

data_path = Path(__file__).parent / "data"
urns = [u for u in data_path.iterdir()]

with open("metadata.jsonl", "r", encoding="utf-8") as f:
    lines = f.readlines()

metadata = {}
for line in lines:
    metadata.update(json.loads(line))

def get_reference_fasta(name):
    filepath = reference_path / names[name]
    if filepath.exists():
        return Fasta(filepath)
    else:
        return None

def get_read_indices(name):
    df = pl.read_csv(
            reference_path / f"{name}.bed",
            separator="\t",
            has_header=False,
            new_columns=["chrom", "start", "end"],
            columns=[0, 1, 2], # Only grab the first 3 columns to save RAM
            dtypes={"chrom": pl.Utf8, "start": pl.Int64, "end": pl.Int64})
    return df


def get_candidate_data(urn, fasta):
    '''Specifically for retrieving the chromosomes associated with Mave DB datasets'''
    chr = metadata[urn]["chr"]
    start = int(metadata[urn]["start"])
    end = int(metadata[urn]["end"])
    seq = fasta[chr_to_refseq[chr]][start - 1:end]
    return seq

@njit
def make_map_fast(chr_map, read_indices):
    for i in range(0, len(read_indices), 2):
        start = read_indices[i]
        end = read_indices[i + 1]
        for j in range(start, end):
            chr_map[j] += 1
    return chr_map / chr_map.max()


class GeneLoader:

    def __init__(self, name, path=None, toprint=False, random_seed=None):

        if sys.platform == "win32":
            reference_path = Path("C:/Users/Eric Arnold/Documents/reference_genome")
        elif 'google.colab' in sys.modules:
            reference_path = Path("/content/drive/MyDrive/nucleoformer/")
        else:
            reference_path = Path("/mnt/c/Users/Eric Arnold/Documents/reference_genome")

        ncbi_path = reference_path / "ncbi_dataset/data/GCF_000001405.40"

        self.idx = 1

        if random_seed is not None:
            np.random.seed=random_seed

        if path is not None:
            ncbi_path = Path(__file__).parent / path
            reference_path = Path(__file__).parent / path

        # self.genome_fasta = get_reference_fasta(name)
        self.genome = pysam.FastaFile(ncbi_path / f"{name}.fa")
        self.bw = pyBigWig.open(str(reference_path / "hg38.phyloP100way.bw"))

        aliases = pl.read_csv(
            reference_path / "chromAlias.txt",
            separator="\t",
            comment_prefix="#",
            has_header=False,
            new_columns=["accession", "UCSCname", "source"]
        )
        aliases = aliases.filter(pl.col("source") == "refseq")
        self.refseq_aliases = dict(zip(aliases["accession"], aliases["UCSCname"]))
        
        self.annot = pl.read_csv(
            ncbi_path / "genomic.gff",
            separator="\t",
            comment_prefix="#",
            has_header=False,
            new_columns=["chrom", "source", "type", "start", "end", "score", "strand", "phase", "attributes"]
        )

        mrna = self.annot.filter(pl.col("type") == "mRNA")
        upstream = 1000
        synthetic_promoters = pl.concat([
            # forward strand
            mrna.filter(pl.col("strand") == "+").with_columns([
                (pl.col("start") - upstream).alias("start"),
                pl.col("start").alias("end"),
                pl.lit("synthetic_promoter").alias("type")
            ]),
            # reverse strand
            mrna.filter(pl.col("strand") == "-").with_columns([
                pl.col("end").alias("start"),
                (pl.col("end") + upstream).alias("end"),
                pl.lit("synthetic_promoter").alias("type")
            ])
        ])

        self.annot = pl.concat([self.annot, synthetic_promoters])

        # if toprint: #save region counts
        #     (self.annot
        #         .with_columns((pl.col("end") - pl.col("start")).alias("span"))
        #         .group_by("type")
        #         .agg(pl.col("span").sum().alias("total_bp"))
        #         .sort("total_bp", descending=True)
        #         .write_csv("type_bp_counts.csv")
        #     )

        chrom_offsets = {}  # chrom -> offset for array filling
        offset_ranges = {}  # range -> chrom for nt_map lookup
        offset = 0
        for chrom in self.genome.references:
            size = self.genome.get_reference_length(chrom)
            chrom_offsets[chrom] = offset
            offset_ranges[range(offset, offset + size)] = chrom
            offset += size
        total_size = offset

        state_array = np.full(total_size, 255, dtype=np.uint8)
        state_encoding = {v: i for i, v in enumerate([k for k in state_dict.keys() if k != "drop"])}

        for state in reversed(list(state_encoding.keys())):
            regions = self.annot.filter(pl.col("type").is_in(state_dict[state]))
            for start, end, chrom in zip(regions["start"].to_list(), regions["end"].to_list(), regions["chrom"].to_list()):
                if chrom not in chrom_offsets:
                    continue
                o = chrom_offsets[chrom]
                state_array[start + o:end + o] = state_encoding[state]

        def in_range(inner, outer):
            return inner.start >= outer.start and inner.stop <= outer.stop

        # state_array = self.merge_nearby_states(state_array, 0, max_gap=24000)

        change_points = np.where(np.diff(state_array))[0]
        starts = np.concatenate([[0], change_points + 1])
        ends = np.concatenate([change_points, [len(state_array) - 1]])
        lengths = ends - starts + 1
        states = state_array[starts]

        chroms = []
        chrom_starts = []
        chrom_ends = []

        for i in range(len(starts)):
            r = range(starts[i], ends[i])
            for offset_range, chrom in offset_ranges.items():
                if in_range(r, offset_range):
                    chroms.append(chrom)
                    offset = offset_range.start
                    chrom_starts.append(starts[i] - offset)
                    chrom_ends.append(ends[i] - offset)
                    break

        self.nt_map = pl.DataFrame({
            "state": states,
            "start": starts,
            "end": ends,
            "len": lengths,
            "chrom": chroms,
            "chrom_start": chrom_starts,
            "chrom_end": chrom_ends
        })

        self.nt_map = self.nt_map.filter(pl.col("state") != 255)

        if toprint:
            print("first chromosome transition")
            for i in range(len(self.nt_map)):
                row = self.nt_map.row(i, named=True)
                global_len = row["end"] - row["start"]
                chrom_len = row["chrom_end"] - row["chrom_start"]
                if global_len != chrom_len:
                    print(self.nt_map[max(0, i-3):min(len(self.nt_map), i+3)])
                    break
            
            for i in range(1, len(self.nt_map)):
                if self.nt_map["chrom"][i] != self.nt_map["chrom"][i-1]:
                    print(self.nt_map.with_row_index()[max(0, i-3):i+3])
                    break

        
        self.state_array = state_array
        self.max_len = len(state_array)
        self.states = list(state_encoding.keys())
        
        if toprint:

            reverse_encoding = {v: k for k, v in state_encoding.items()}
            ids, counts = np.unique(state_array, return_counts=True)
            unique_names = [reverse_encoding.get(i, "Unknown") for i in ids]
            pdict = dict(zip(unique_names, counts))
            total = sum(pdict.values())

            print(f"{'Name':<20} {'Count':>8} {'Percentage':>10}")
            print("-" * 40)
            for key, value in pdict.items():
                pct = (value / total) * 100
                print(f"{key:<20} {int(value):>8} {pct:>9.1f}%")
            print("-" * 40)
            print(f"{'TOTAL':<20} {int(total):>8} {'100.0':>9}%")

            x = self.nt_map.with_columns(
                pl.col("state").shift(-1).alias("next_state")
            ).group_by(["state", "next_state"]).len().sort("len")
            x = x.with_columns([
                pl.col("state").map_elements(lambda s: reverse_encoding.get(s, "null")).alias("state_name"),
                pl.col("next_state").map_elements(lambda s: reverse_encoding.get(s, "null")).alias("next_state_name")
            ])
            with pl.Config(tbl_rows=100, tbl_cols=20):
                print(x)


    def merge_nearby_states(self, state_array, state_code, max_gap=4000):
        """Merge regions of state_code that are within max_gap bp of each other."""
        is_state = (state_array == state_code)
        
        # find start/end of each state region
        padded = np.concatenate([[False], is_state, [False]])
        starts = np.where(~padded[:-1] & padded[1:])[0]
        ends = np.where(padded[:-1] & ~padded[1:])[0]
        
        if len(starts) == 0:
            return state_array
        
        # merge regions within max_gap of each other
        merged_starts = [starts[0]]
        merged_ends = []
        for i in range(1, len(starts)):
            if starts[i] - ends[i-1] <= max_gap:
                # gap is small enough, extend current region
                pass
            else:
                merged_ends.append(ends[i-1])
                merged_starts.append(starts[i])
        merged_ends.append(ends[-1])
        
        # fill gaps between merged regions
        for s, e in zip(merged_starts, merged_ends):
            state_array[s:e] = state_code
        
        return state_array


    def get_idx(self, idx=None, with_row=False):
        if idx == None:
            idx = np.random.randint(self.nt_map.shape[0])
        row = self.nt_map.row(idx, named=True)
        x = 40 if row["len"] < 20 else row["len"]
        max_L = self.genome.get_reference_length(row["chrom"])
        region = round(x * 0.1)
        start = max(0, row["chrom_start"] - region)
        end = min(max_L, row["chrom_end"] + region + 1)
        seq = self.genome.fetch(row["chrom"], start, end)
        # annot = self.annot.filter(pl.col("chrom") == row["chrom"])
        ucsc_key = self.refseq_aliases[row["chrom"]]
        conservation = self.bw.values(f"{ucsc_key}", start, end)

        offset = row["start"] - row["chrom_start"]

        start = max(offset, row["start"] - region)
        end = min(offset + max_L, row["end"] + region + 1)
        states = self.state_array[start: end]
        if with_row:
            entry =  {"seq": seq, "conv": np.array(conservation), "states": states}
            for key in row:
                entry[key] = row[key]
            return entry
        return {"seq": seq, "conv": conservation, "states": states}

    def get_jobs(self, n_jobs=16, batch_size=1000):
        jobs = []
        for job in range(n_jobs):
            jobs.append(self[self.idx: self.idx + batch_size])
            self.idx += batch_size
        return jobs

    def reset_idx(self):
        self.idx=1
    
    def __len__(self):
        return self.nt_map.shape[0]

    def __getitem__(self, index):
        if isinstance(index, slice):
            items = [self.get_idx(i) for i in range(*index.indices(len(self)))]
            return {key: [item[key] for item in items] for key in items[0]}
        return self.get_idx(index)
    


        






        


    


    


    
if __name__ == "__main__":
    picked = []
    for urn in metadata:
        if "reference" in metadata[urn]:
            if metadata[urn]["reference"] == "hg19":
                picked.append(urn)

    genome = get_reference_fasta("hg19")
    seq = get_candidate_data(picked[0], genome)
    response = requests.get(f"https://api.mavedb.org/api/v1/score-sets/{picked[0]}")

    print(f"=== MaveDB Validation Check ===")
    print(f"\n[MaveDB API Response] Score set data for URN '{picked[0]}':")
    print(response.json())

    print(f"\n[Local Retriever] Candidate sequence for URN '{picked[0]}':")
    print(seq)

    print(f"\n=== End Validation ===")