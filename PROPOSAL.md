# Project Proposal


## Concept:

Genome language models (gLMs) (e.g. DNABert, DNABert2, NucleotideTransformer) have been developed as tunable platforms for genomic tasks, such as classification and prediction. While these models perform reasonably on tasks involving genomes within or adjacent to their training data, the brittleness and generalizability of their knowledge remains an open question. As many models use similar algorithms for encoding and processing (e.g. bytepair encoding and sentencepiece tokenization in DNABert2) and identical transformer architecture and pretraining strategies, such questions in the realm of genomics reflect broader concerns with LLMs, chiefly whether they are sophisticated pattern recognizers or if abstract and generalizable knowledge is encoded in their latent space [6]. 

Recently, researchers have developed synthetic tests that examine the robustness [2] and generalizability [1] of gLMs. One limitation of these studies is the plausibility of their sequences. The Nullsettes method generates chimeric sequences by rearranging regulatory elements in casettes from massive parallel reporter assay datasets. The researchers find that gLMs generally "perform poorly on zero-shot [loss of function] prediction, particularly under contexts that deviate strongly from natural sequence statistics." An alternative approach is the perturberation method, which mutates sequences using NLP-derived methods (backgranslation and embedding-level changes). The plausibility of mutant sequences is then evaluated with NCBI BLAST (species recognition rate and mapping rate) and "CG content." While such sequences are hypothetically more realistic than Nullsettes, the researchers report mutation rates ranging from 10-50%, far greater than disease induced variation within a species. Furthermore, the issue of context window size and numerical precision [7] is not discussed in relation to their findings, which is particularly relevant to models using ALiBi [3] to achieve unlimited context, where the influence SNPs might become undetectable at a scale necessary for some analysis tasks, such as analysis where elements might be separated by thousands of bases.

Thus, this study sets out to do the following:
- Address the limitations of the aformentioned studies by evaluating the impact of DMS mutations on regulatory classification at different context window sizes and levels of quantization
    - Sources: MaveDB, VAMP-seq
- Use a genetic algorithm in combination with a PWM trained on the relevant reference genome to generate reasonable perturberations (in reasonable proportion) that sketch the boundary of model perception
    - This method can be used to assesss both model generalizability and its sensitivity to clinical genetic variance.
    - The method would target specifically the flanking regions (chiefly spacers) of promoters etc. To test the independence of classifications from context.
- Examine the hidden layer to see what the model is paying attention to
- Synthesize these findings with similar models using natural language sequences.


## Bibliogrpahy
[1] Nulsettes: https://arxiv.org/abs/2506.10271v3
[2] Robustness: https://arxiv.org/abs/2409.19788v2
[3] DNABert2: https://arxiv.org/abs/2306.15006
[4] Nucleotide transformer: https://www.nature.com/articles/s41592-024-02523-z
[5] Review: https://arxiv.org/abs/2412.07201
[6] Keplerian vs newtonian understanding: https://arxiv.org/abs/2111.12210
[7] Transformers need glasses!: https://arxiv.org/abs/2406.04267

#AI Appendix

Gemini 3: should a genome analysis model like DNABert2 be able to correctly identify a promoter region even if it is surrounded by mutations?
Claude 4.5:
- Would you say that the mutations in DMS datasets are more "legitimate" than random mutations?
- I just have scores
- Right, I'm comparing this to synthetically generated sequences with random mutations
- Well I was going to use synthetic sequences to fool the model. A lot of these mutations are single nucleotide, so I can pretty much rule out the model seeing them in many cases
- Hmm. DNABert2 wasn't trained on DMS data. I was originally just going to use mutations to see how its region classifications change. But maybe if I did train it on DMS and then subjected it to random mutations, that might be a robust test of its discriminatory ability. I was just going to take the pretrained model and see how robust its classifications were
- how big is a chromosome? Would it make sense to find all the promoters within a chromosome?
- unless the mutations are in the promoter region, should we expect the model to classify promoters?
- I have no idea. My question is if the model should from a biological theory standpoint be able to recognize the promoter despite surrounding mutations. Like does this make sense
- Well something else that they don't address is how much perturberation fools the model. At some point a gene/promoter/sequence should become unrecognizable. So the question is how mutated. I don't really think they address this in the paper. Is the amount of mutation "reasonable" for how a patient might present in a clinical application?

etc.
















