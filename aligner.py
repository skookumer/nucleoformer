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
    "CDS": ["CDS"],
    "promoter": ["promoter", "TATA_box", "CAAT_signal", "GC_rich_promoter_region", "CAGE_cluster"],
    "enhancer": ["enhancer"],
    "exon": ["exon", "mRNA", "primary_transcript", "transcript", "pseudogene"],
    "ncRNA": ["lnc_RNA", "miRNA", "tRNA", "rRNA", "snRNA", "snoRNA", "scRNA", "scaRNA", 
              "Y_RNA", "RNase_MRP_RNA", "RNase_P_RNA", "telomerase_RNA", "vault_RNA", "antisense_RNA"],
    "repeat": ["microsatellite", "minisatellite", "tandem_repeat", "repeat_region", 
               "mobile_genetic_element", "dispersed_repeat", "direct_repeat", "repeat_instability_region"],
    "regulatory": ["insulator", "silencer", "locus_control_region", "imprinting_control_region",
                   "enhancer_blocking_element", "regulatory_region", "transcriptional_cis_regulatory_region",
                   "response_element", "protein_binding_site", "matrix_attachment_site",
                   "DNaseI_hypersensitive_site", "epigenetically_modified_region"],
    "immune": ["V_gene_segment", "D_gene_segment", "J_gene_segment", "C_gene_segment"],
    "drop": ["gene", "region", "biological_region", "match", "cDNA_match", "sequence_comparison",
             "sequence_feature", "sequence_alteration", "sequence_alteration_artifact",
             "chromosome_breakpoint", "nucleotide_cleavage_site", "nucleotide_motif",
             "sequence_secondary_structure", "conserved_region", "centromere", "D_loop",
             "origin_of_replication", "replication_start_site", "replication_regulatory_region",
             "meiotic_recombination_region", "mitotic_recombination_region",
             "non_allelic_homologous_recombination_region", "recombination_feature",
             "nucleotide_motif", "nucleotide_cleavage_site"]
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
        # self.genome = pysam.FastaFile(ncbi_path / f"{name}.fa")
        
        self.annot = pl.read_csv(
            ncbi_path / "genomic.gff",
            separator="\t",
            comment_prefix="#",
            has_header=False,
            new_columns=["chrom", "source", "type", "start", "end", "score", "strand", "phase", "attributes"]
        )
        
        # .with_columns([
        #     (pl.col("start") - 1).alias("start"),
        #     (pl.col("end") - 1).alias("end"),
        # ])


        aliases = pl.read_csv(
            reference_path / "chromAlias.txt",
            separator="\t",
            comment_prefix="#",
            has_header=False,
            new_columns=["accession", "UCSCname", "source"]
        )
        aliases = aliases.filter(pl.col("source") == "refseq")
        self.refseq_aliases = dict(zip(aliases["accession"], aliases["UCSCname"]))
        self.bw = pyBigWig.open(str(reference_path / "hg38.phyloP100way.bw"))
    
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