Please see hmmer.ipynb for a demonstration

This program uses psyam for the fasta file and pybigwig for the phyloP conservation scores. These files total about 30gb so I have not included them.

For the transformers library, you will need a specific version to run the model locally. This was a very complicated and fragile process to get it to run on CPU for debugging. I'm therefore not providing requirements.

# Progress Report
## Research Question

I've reduced the scope back to model testing. It is simply not feasable to train or fine-tune a model in the time remaining. So, this program is intented to test DNABert to see if it has learned conservation and to understand the brittleness of its representations.

## Algorithm

The algorithm has 3 main components.
- A loader supplies the genome, annotations, and phylop conservation scores.
- The HMM provides locally-relevant nucleotide mutations and can "heal" or smooth over areas where mutations have taken place.
- The genetic algorithm (NSGA-II) finds the range of optimal solutions that balance the objectives, such as hamming distance, levenstein, HMM probability, etc. against cosine distance of the transformer embeddings to the original

## Current implementation status

Everything is mostly working. I have a few things to debug before throwing it onto colab and generating exhaustive dataframes under multiple objectives. Then I need to take the results under those objectives and compare them in aggegrate. So the statistical analysis portion is what remains.

# What's implemented

As stated above, the three parts of the program are implemented. There are some deviations from the pseudocode but honestly I don't remember what they are at this point. The healing/smoothing mecahnism is working.

# Prototype Demo

## Minimal prototype run

See hmmer.ipynb. For each population, it produces parquet files that can be used for downstream analysis.

## Input

The program expects a fasta file, GFF annotations, phylop scores all associated with the same genome

## Output

Parquet files

## Example

See hmmer.ipynb

# Data Documentation

- The test data is HG38 from NCBI. Had to download the whole 30gb set + annotations. PhyloP is from UCSC. Had to also get the labelling scheme to align the chromosome calls between the two databases.
- Split the genome according to the GFF annotations so that the HMM has distinct functional states.
- The original data is the ground truth. The point is to measure the deviation of the DNABert model wrt to ground truth. So Truth is built into the process.

# Initial Observations

- Yes; I just need to debug reversing objectives.
- No surprising behavior. Getting njit lists to run was a pain in the ass. Getting DNABert to run on CPU when it's hardcoded in Triton was a pain in the ass.
- No unexpected output.
- Preliminary performance: Fast

# Divergence:

The plan diverged in many ways. Huge challenges in aligning everything. Put more time into it to make the project manageable.

# Next steps

Need to collect Data

# Gen AI

Lots of gen ai used. I didn't understand HMMs initially so I did an initial vibe pass so I had something to test with. I've since gone back and refactored. Lots of AI used to get the bio stuff working together and DNABert.
