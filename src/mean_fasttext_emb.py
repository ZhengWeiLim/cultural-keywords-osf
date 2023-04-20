import io
import os
import numpy as np
import json
import argparse
from util import load_lemmatized

parser = argparse.ArgumentParser()
parser.add_argument("--aligned_vectors_dir", default=None, type=str, required=True,
                    help="fasttext aligned vectors directory")
parser.add_argument("--lemma_dir", default=None, required=True,
                    help="words and lemma directory")
parser.add_argument("--lang", default=None, type=str,
                    help="language iso")
parser.add_argument("--output_dir", default=None, type=str,
                    help="mean vectors output directory")
args = parser.parse_args()


fasttext_emb_file = os.path.join(args.aligned_vectors_dir, "wiki.{}.align.vec".format(args.lang))
save_emb_file = os.path.join(args.output_dir, "{}.json".format(args.lang))
word_lemma_file = os.path.join(args.lemma_dir, "{}.txt".format(args.lang))


def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        word = tokens[0].lower()
        emb = list(map(float, tokens[1:]))
        if word in data:
            data[word] = np.concatenate((data[word], np.expand_dims(np.double(emb), axis=0)))
        else:
            data[word] = np.expand_dims(np.double(emb), axis=0)

    for word, dt in data.items():
        data[word] = np.mean(data[word], axis=0)

    return data

word_lemma = load_lemmatized(word_lemma_file)
word_vectors = load_vectors(fasttext_emb_file)

lemma_words = {}
for wd, lm in word_lemma.items():
    lemma_words[lm] = lemma_words.get(lm, [])
    lemma_words[lm].append(wd)

lemma_vectors = {}
for lemma, words in lemma_words.items():
    vectors = [word_vectors[w] for w in words if w in word_vectors]
    if vectors:
        lemma_vectors[lemma] = np.mean(vectors, axis=0)


vectors_jsonified = {lemma: v.tolist() for lemma, v in lemma_vectors.items()}
with open(save_emb_file, 'w') as f:
    json.dump(vectors_jsonified, f)

print("saved mean lemma vectors at {}".format(save_emb_file))