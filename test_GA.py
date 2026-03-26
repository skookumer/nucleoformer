from aligner import get_reference_fasta, GeneLoader
from genetic_algorithm import NSGA_II
from HMM import HMM
import numpy as np
from pprint import pprint

#curl -o datasets 'https://ftp.ncbi.nlm.nih.gov/pub/datasets/command-line/v2/linux-amd64/datasets'
#chmod +x datasets

#./datasets download genome accession GCF_000001405.40 --include genome,gff3 --chromosomes 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,X,Y


# # 1. Unzip the package
# unzip ncbi_dataset.zip

# # 2. Move into the data folder
# cd ncbi_dataset/data/GCF_000001405.40/

# # 3. Merge the 24 chromosome files into one "Primary Assembly"
# cat chr1.fna chr2.fna chr3.fna chr4.fna chr5.fna chr6.fna chr7.fna chr8.fna chr9.fna chr10.fna chr11.fna chr12.fna chr13.fna chr14.fna chr15.fna chr16.fna chr17.fna chr18.fna chr19.fna chr20.fna chr21.fna chr22.fna chrX.fna chrY.fna > hg38_primary.fa

# # 4. Create the 'Index' so Python/Pysam can jump around the 3GB file instantly
# samtools faidx hg38_primary.fa



LOADER = GeneLoader("hg38_primary")
# print(loader.get_idx(1))
# print(loader.get_idx(0))
# loader.test(loader.nt_map.shape[0] - 1)
# loader.test(2)
MODEL = HMM(n_states=len(LOADER.states), k=6, n_bases=6, model_name="supervised_10k_oldschema")
objectives = {
    "cos_dist": {"reverse": False},
    "ham_dist": {"reverse": False},
    "hmm_prob": {"reverse": False}
}
GA = NSGA_II("test", objectives, MODEL, LOADER, pop_cap=2, idx=21)
GA.load_popn()
with np.printoptions(threshold=200, edgeitems=3):
    for i in range(50):
        GA.random_mutate_entry(0)
        GA.random_mutate_entry(1)
    GA.crossover(0, 1, 2)
    print(GA.popn)

