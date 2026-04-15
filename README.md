# nucleoformer
evaluation suite for nucleotide transformer models

See the presentation poster PDF for some results. I tried exporting the analysis notebook from colab but it unfortunately removes all the figures. you can view it here:

https://colab.research.google.com/drive/1P-IQJ0G61caOJ13-Lp3JopQOR3rwokVv

# QUICK START

## Project overview

### Summary:

This is an adversarial testing program designed for probing genomic language model understanding of sequences. It is intended to measure the degree to which the model has learned conservation in a given species and how robust its representations are to deviation.

To do this, the program uses a genetic algorithm (GA) NSGA-II for multi-objective optimization and a first-order Hidden Markov Model (HMM) for mutation. Furthermore, a separate loader class is used to transform annotations into states for the HMM to use. Finally, there is the model being tested, in this case Nucleotide Transformer. The GA coordinates operations between all modules and progressively evolves genetic sequences in a manner that maximizees (or minimizies) the given objectives.

Three are four types of data used in this project: The loader class provides and aligns NCBI GFF annotations with genetic sequences also from NCBI. Additionally, UCSC phyloP scores are aligned with these annotated sequences. The final source of data are the embeddings produced by the genomic language model. These are combined by the GA across iterations.

### Installation/setup:

You can attempt to run this locally with a professional-grade graphics card by installing the required packages and obtaining the necessary data (see section below). Alternatively, you can use the Colab notebook and use a google colab server. This is the preferred method. In this case, you may just set up your google drive directory accordingly and run the cell in the notebook which installs the proper packages for running on colab.

Drive directory: Create a folder in your drive and include it in the path within the mount google drive cell. Within this path, you will need to place the main scripts genetic_algorithm.py, aligner.py, and HMM.py. Include the Fasta file and GFF annotations in a "genome" folder.

Python version: 3.11.14 allows for the use of both transformers 4.52 and pysam/pybigwig. Version 3.12 on Colab also works.

If running locally, ensure that your graphics card has at least 16gb of vram to host the model and any sequences. You might want to tune the GA batch_size parameter or just set it to 1 since you are operating on limited memory.

## Quick Start:

Colab: just run all the cells in sequence. These install the necessary packages and call the scripts from the mounted location. You might need to mount your drive to the proper location. See Google Drive documentation for guidance on this.

## Data Collection:

I dowloaded my data directly from UCSC golden path and NCBI. The key was to download all NCBI files including the gff and then combine them into a single document.

HG38 + annotations (linux terminal):

```python
curl -o datasets 'https://ftp.ncbi.nlm.nih.gov/pub/datasets/command-line/v2/linux-amd64/datasets'
chmod +x datasets

./datasets download genome accession GCF_000001405.40 --include genome,gff3 --chromosomes 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,X,Y

1. Unzip the package
unzip ncbi_dataset.zip

2. Move into the data folder
cd ncbi_dataset/data/GCF_000001405.40/

3. Merge the 24 chromosome files into one "Primary Assembly"
cat chr1.fna chr2.fna chr3.fna chr4.fna chr5.fna chr6.fna chr7.fna chr8.fna chr9.fna chr10.fna chr11.fna chr12.fna chr13.fna chr14.fna chr15.fna chr16.fna chr17.fna chr18.fna chr19.fna chr20.fna chr21.fna chr22.fna chrX.fna chrY.fna > hg38_primary.fa

4. Create the 'Index' so Python/Pysam can jump around the 3GB file instantly
samtools faidx hg38_primary.fa
```
UCSC PhyloP scores:

```python
go to: https://hgdownload.soe.ucsc.edu/goldenPath/hg38/phyloP100way/

Navigate down to hg38.100way.phyloP100way/
and click it
```

## Usage and options:

Main scripts:

- GeneLoader (aligner.py): This script aligns GFF annotations, phyloP scores, and nucleotides themselves. Returns a fixed window that overlaps state transitions by 10% of central sequence length.

- NSGA-II (genetic_algorithm.py): This is the main coordinator algoirthm that initializes populations, handles mutation and crossover, calls the HMM for mutations and healing mutations, and calls the genomic model for embeddings. Has options for population size, objective type and direction, batch_size for embeddings. Can use hamming distance, HMM probability, cosine distance, conservation score, conservation deviation, and Levenshtein distance as metrics.

- HMM (HMM.py): A first-order hidden markov model with k-merization. Facilitates prefixed-generation and healing from the genetic algorithm. Currently is just trained in a supervised fashion. The number of states should be matched with those of the GeneLoader it is associated with. Uses pretrained matrices from a probability store which are included with the repo for reproducability.

## Limitations:

This program is designed to do one thing, and it's best used in colab. It can be adapted to Slurm with a custom script following the format of the notebook. There is no provision for logging or for use with other HMM training methods such as Baum-Welch for unsupervised training. The GA's code for getting embeddings might be limited to Nucleotide Transformer due to incompatibilities with those models.

It is currently assumed that you are running this program on colab with Nucleotide Transformer with the provided notebook and the specific data downloaded from NCBI and UCSC.

Scalability: using the model gradients to target nucleotide mutation has dropped the number of iterations required to roughly 50% of the original, or 250 for most sequences < 2000bp long. Nevertheless, the algorithm completes roughly 15 sequences per hour on a single thread, which is a small number of sequences for an entire genome. The main bottleneck to performance is the embedding step since so many need to be generated. Perhaps in the future a linear approximation could be used to reduce the number of embeddings required.

## Evidence of Correctness:

It's difficult to say that a genetic algorithm is producing a "correct" result. Empirically, we can observe a pareto front forming between the objectives. A few tests have been included in old_scripts that test HMM output. The suite correctly identifies weak spots and exhibits variation in an expected manner across different objectives.

## Final Reflection:

### What went right:

The project was a success in my view. This is a fairly big program and I spent about a month ahead of time writing and debugging everything to ensure that I got the results on time. The GFF data provided an appropriate amount of information to train an HMM and to find meaningful differences in the embedding model's representations (though after debiasing).

### Challenges:

It was difficult to approach the problem biologically and to find datasets that would allow me to meaningfully test the model. I had initially planned on using DMS data, but this turned out to be far too sparse for what I wanted to do. I have since learned that clinvar datasets have mutated variants of the human genome, so I plan on testing those.

Another issue was training the HMM. I had initially wanted to use Baum-Welch with a two-layered model where state-patterns were emissions for a nucleotide model. This didn't work out because I didn't get Baum-Welch working in time. I think the code is quite close and I will have to review it again to figure out what I want to do in this regard.

## Future directions:

I want to incorporate clinvar data into the testing as a baseline. Regarding mutations, I need to incorporate the debiased PCA model into the cosine distance calculation as the raw vectors are generating distances that are not really representative of the space each annotated class is in. To improve the HMM model, I want to get baum-welch working and intorduce more disruptive mutations via hierarchical-hidden-state-vocabulary swapping. I also want to create indel methods for the GA and see how varying sequence length affects the model's representation. Finally, it seems like this could be a good model for RTD training, so perhaps I will fine-tune a version of NT using this method and see how it pans out.

## Gen AI use:

This project wouldn't have been possible without generative AI in the amount of time I had to complete it. Areas of particularly strong use include using the model gradient to compute sensitivities. I haven't done this before and really have no insight into how to use pytorch to do this. Another area of usage was generating the statistical tests for differences in annotation categories and learning about debiasing. I've never encountered this issue before so it was an opportunity to understand why the pre-trained model was cramming everything into a small part of the embedding space and how to mitigate that. Lots of AI use throughout for debugging etc. 
