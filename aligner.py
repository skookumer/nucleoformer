import polars as pl
import json
from pathlib import Path
from pyfaidx import Fasta
import requests
from pprint import pprint

chr_to_refseq = {
    '1': 'NC_000001.10',   '2': 'NC_000002.11',   '3': 'NC_000003.11',
    '4': 'NC_000004.11',   '5': 'NC_000005.9',    '6': 'NC_000006.11',
    '7': 'NC_000007.13',   '8': 'NC_000008.10',   '9': 'NC_000009.11',
    '10': 'NC_000010.10',  '11': 'NC_000011.9',   '12': 'NC_000012.11',
    '13': 'NC_000013.10',  '14': 'NC_000014.8',   '15': 'NC_000015.9',
    '16': 'NC_000016.9',   '17': 'NC_000017.10',  '18': 'NC_000018.9',
    '19': 'NC_000019.9',   '20': 'NC_000020.10',  '21': 'NC_000021.8',
    '22': 'NC_000022.10',  'X': 'NC_000023.10',   'Y': 'NC_000024.9',
    'MT': 'NC_012920.1'
}

reference_path = Path("/mnt/c/Users/Eric Arnold/Documents/reference_genome")
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

def get_candidate_data(urn, fasta):
    chr = metadata[urn]["chr"]
    start = int(metadata[urn]["start"])
    end = int(metadata[urn]["end"])
    seq = fasta[chr_to_refseq[chr]][start - 1:end]
    return seq
    


    
if __name__ == "__main__":
    picked = []
    for urn in metadata:
        if "reference" in metadata[urn]:
            if metadata[urn]["reference"] == "hg19":
                picked.append(urn)

    genome = get_reference_fasta("hg19")
    seq = get_candidate_data(picked[0], genome)
    response = requests.get(f"https://api.mavedb.org/api/v1/score-sets/{picked[0]}")
    print(response.json())
    print(picked[0])

    print(seq)