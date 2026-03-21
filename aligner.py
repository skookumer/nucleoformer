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

state_dict = {
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

    def __init__(self, name):

        # self.genome_fasta = get_reference_fasta(name)
        self.genome = pysam.FastaFile(ncbi_path / f"{name}.fa")

        aliases = pl.read_csv(
            reference_path / "chromAlias.txt",
            separator="\t",
            comment_prefix="#",
            has_header=False,
            new_columns=["accession", "UCSCname", "source"]
        )
        aliases = aliases.filter(pl.col("source") == "refseq")
        self.refseq_aliases = dict(zip(aliases["accession"], aliases["UCSCname"]))
        
        annot = pl.read_csv(
            ncbi_path / "genomic.gff",
            separator="\t",
            comment_prefix="#",
            has_header=False,
            new_columns=["chrom", "source", "type", "start", "end", "score", "strand", "phase", "attributes"]
        )

        chrom_offsets = {}  # chrom -> offset for array filling
        offset_ranges = {}  # range -> chrom for nt_map lookup
        offset = 0
        for chrom in self.genome.references:
            size = self.genome.get_reference_length(chrom)
            chrom_offsets[chrom] = offset
            offset_ranges[range(offset, offset + size - 1)] = chrom
            offset += size
        total_size = offset

        state_array = np.full(total_size, 255, dtype=np.uint8)
        state_encoding = {v: i for i, v in enumerate([k for k in state_dict.keys() if k != "drop"])}

        for state in reversed(list(state_encoding.keys())):
            regions = annot.filter(pl.col("type").is_in(state_dict[state]))
            for start, end, chrom in zip(regions["start"].to_list(), regions["end"].to_list(), regions["chrom"].to_list()):
                if chrom not in chrom_offsets:
                    continue
                o = chrom_offsets[chrom]
                state_array[start + o:end + o] = state_encoding[state]

        def in_range(inner, outer):
            return inner.start >= outer.start and inner.stop <= outer.stop

        state_array = self.merge_nearby_states(state_array, 0, max_gap=2000)

        change_points = np.where(np.diff(state_array))[0]
        starts = np.concatenate([[0], change_points])
        ends = np.concatenate([change_points - 1, [len(state_array) - 1]])
        lengths = ends - starts
        states = state_array[starts]

        chroms = []
        for i in range(len(starts)):
            r = range(starts[i], ends[i])
            for offset_range, chrom in offset_ranges.items():
                if in_range(r, offset_range):
                    chroms.append(chrom)
                    break

        self.nt_map = pl.DataFrame({
            "state": states,
            "start": starts,
            "end": ends,
            "len": lengths,
            "chrom": chroms
        })

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

            

        
        # .with_columns([
        #     (pl.col("start") - 1).alias("start"),
        #     (pl.col("end") - 1).alias("end"),
        # ])

        self.bw = pyBigWig.open(str(reference_path / "hg38.phyloP100way.bw"))

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


        


    def test(self):



        key = self.genome.references[0]
        seq = self.genome.fetch(key)
        annot = self.annot.filter(pl.col("chrom") == key)
        ucsc_key = self.refseq_aliases[key]
        conservation = self.bw.values(f"{ucsc_key}", annot["start"][0], annot["end"][0])
        print(seq[:20])
        print(annot)
        print(conservation)


        






        


    


    


    
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