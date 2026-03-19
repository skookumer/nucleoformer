from aligner import get_reference_fasta, GeneLoader

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
loader = GeneLoader("hg38_primary")
# loader.test()