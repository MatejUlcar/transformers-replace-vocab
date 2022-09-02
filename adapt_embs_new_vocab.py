from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM
import torch
import argparse

""" This script loads a transformer model and replaces its embedding layer with a new one. The new embeddings are initialized from the old ones.
    Where the tokens are same, the weights are kept, where they are not, the new weights are set as the average of the subtokens that make up the token.
    Current limitation: the special token (eos, bos, unk, pad, mask) embeddings do not get copied correctly.
    Adapted from: Koto et al. IndoBERTweet. (2021). https://arxiv.org/abs/2109.04607v1
"""

parser = argparse.ArgumentParser(description='Replace tokenizer and embeddings layer in a transformer with new ones')
parser.add_argument('-m', '--model', default='.', help='huggingface ID or path to the model you are adapting')
parser.add_argument('-o', '--output', help='where to save the newly adapted model')
parser.add_argument('-d', '--dictionary', default='dict.txt', help='path to the new vocabulary/dictionary file')
parser.add_argument('-s', '--skip', default=3, help='how many initial lines to skip in dict.txt')
args = parser.parse_args()

og_model = args.model
t = AutoTokenizer.from_pretrained(og_model)
m = AutoModel.from_pretrained(og_model)

newvocab = []
new_e = []

# load new vocab/dict
with open(args.dictionary, 'r') as f:
    for line in f:
        line = line.split()
        newvocab.append(line[0])


# iterate over new vocab, produce new embs as avg of og_model embs
for i,x in enumerate(newvocab):
    if i < args.skip: # skip special tokens unk,bos,eos
        continue
    c = t.encode(x, return_tensors='pt')
    c = c[:,1:-1] # remove bos and eos tokens
    e = m.embeddings.word_embeddings(c)
    e = e.mean(axis=1)[0]
    new_e.append(e.data)


# set new embs to the model
m.resize_token_embeddings(len(newvocab)+5) # allow for +5 specials: eos,bos,unk,pad,mask
with torch.no_grad():
    for i in range(len(new_e)):
        vocab_index = i+args.skip # first (0) element in new_e is 0+3=fourth element in dict.txt
        embed_index = vocab_index+4 # starts with 4 special tokens (eos,bos,pad,unk)
    m.state_dict()['embeddings.word_embeddings.weight'][embed_index] = torch.tensor(new_e[i])
    

# now save model as new model
m.save_pretrained(args.output)
