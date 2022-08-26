from transformers import AutoTokenizer, AutoModel
import torch
import argparse

""" This script loads a transformer model and replaces its embedding layer with a new one. The new embeddings are initialized from the old ones.
    Where the tokens are same, the weights are kept, where they are not, the new weights are set as the average of the subtokens that make up the token.
    Current limitation: the new vocabulary size must be equal to or smaller than the old one.
    Adapted from: Koto et al. IndoBERTweet. (2021). https://arxiv.org/abs/2109.04607v1
"""

parser = argparse.ArgumentParser(description='Replace tokenizer and embeddings layer in a transformer with new ones')
parser.add_argument('-m', '--model', help='huggingface ID or path to the model you are adapting')
parser.add_argument('-o', '--output', help='where to save the newly adapted model')
parser.add_argument('-d', '--dictionary', default='dict.txt', help='path to the new vocabulary/dictionary file')
args = parser.parse_args()

og_model = args.model #'ai4bharat/indic-bert'
t = AutoTokenizer.from_pretrained(og_model)
m = AutoModel.from_pretrained(og_model)
m.eval()
new_e = []
newvocab = []

# load new vocab/dict
with open(args.dictionary, 'r') as f:
    for line in f:
        line = line.split()
        newvocab.append(line[0])

# iterate over new vocab, produce new embs as avg of og_model embs
for x in newvocab:
    c = t.encode(x, return_tensors='pt')
    c = c[:,1:-1] # remove <s> and </s> tokens
    e = m.embeddings.word_embeddings(c)
    new_e.append(e.mean(axis=1))

# set new embs to the model
with torch.no_grad():
    for i in range(len(new_e)):
        m.embeddings.word_embeddings.weight[i] = new_e[i]
m.embeddings.word_embeddings.weight = torch.nn.Parameter(m.embeddings.word_embeddings.weight[:len(newvocab)+5])
m.embeddings.word_embeddings.num_embeddings = len(newvocab)+5

# now save m as new model
m.save_pretrained(args.output)
