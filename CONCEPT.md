# Research Question

We've had a little scope creep since the last assignment. The core algorithm remains the same, but the evaluation method has changed. Instead of measuring the difference between classification labels for the BERT model, I'll be computing the cross entropy loss across the tokens to determine the genetic algorithm's fitness function. We're still using an MCMC method to guide the mutation of the sequence.

The big new change is that I'll be training a BERT model on the same training data as DNABert2, but with an replaced token method (RTD). This more advanced method has been in NLP for a while, but has only been recently applied to genomic models (NucEL). Expanding RTD to nucleotide transformers is an interesting challenge as it requires a somewhat sophisticated generator model to produce the replaced tokens. The goal here is to test traditional methods in contrast with DNABert2 and NucEL.

So, there are now two parts:
- Evaluate gLMs using the genetic algorithm + MCMC + HMM method using cross-entropy loss
- Flip the script and train a gLM using MCMC + HMM

The shift to training a model is significant. Genomic language models offer NLP an interesting ability to evaluate the models on a completely different set of data, which may reveal weaknesses/strengths that are underemphasized by natural language.

# Inputs/outputs:

## Testing:
The testing algorithm will use nucleotide sequences with read indices obtained for HG19 and HG38. These will be downsampled using KDE to ensure that runtime with MCMC is viable. The transformer model will be fed a certain context, and its output embedding will be compared against its dictionary for loss. The genetic algorithm will progressively mutate the sequence and catalog loss along the way.

## Training:
This is a tranformer pre-training pipeline. You provide input sequences with a data loader, replace the tokens using the RTD method, get the model's output vector, and compute loss, and call backpropagation to modify the weights. After you do this for a certain number of epochs, the model produces meaningful bidirectional embeddings that can be used for downstream tasks.

# Pseudocode:
```

def gen_sequence(reference, epsilon=x):
    let PWM be the weight matrix of the reference genome
    let HMM be the heirarchical markov model from the reference genome
    seq, HMM_score = smoothed_HMM.generate()
    PWM_seq = GibbsMotifSampler(seq)
    JSD = jsd(PWM, PWM_seq)
    Ham_d = hamming_dist(seq, reference)
    while JSD > epsilon:
        tighten_sequence(JSD, Ham_d)
    return seq, JSD, PWM_seq, Ham_d, HMM_score
    
Genetic algorithm:

seqs = gen_sequences(epsilon)
model = load_bert_model()

for iteration in max_iter:
    for seq in seqs:
        prediction = model.predict(seq)
        obj_A = ensemble_score(JSD, Ham_d, HMM_score)
        obj_B = cross_entropy(prediction)
        obj_c = Enformer functional score (maybe) or just Ham_d, or something to do with flattened attention

Train Transformer:
    input = data_loader.load()
    input = smooted_HMM.generate()
    loss = model.predict(input)
    model.backpropagate(loss)

```

Complexity is bad. I'll have to figure out how many reads to keep while maximizing statistical significance.

# Pitfalls:

I'm biting off a big chunk by trying to train a model on top of everything else. I'll know how feasable this is once I get the genetic algorithm up and running and computing loss. I'm planning on using Explorer (if my credentials are still active) or kagglehub to do the training. Things here are kind of vague, but I'm trying to do something somewhat original with this project.
