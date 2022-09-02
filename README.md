# transformers-replace-vocab

Replace the vocabulary and embeddings layer of a transformers model in pytorch format. The new embeddings are initialized from the existing embedding weights. For example, if a new token already exists in the old vocabulary, its embedding weights are copied. If a new token is not in the old vocabulary, it is split into subtokens that are in the old vocabulary. The new token embeddings are then the average of its subtoken embeddings.

The special token (bos, eos, pad, unk, mask) weights are not yet correctly copied over.
