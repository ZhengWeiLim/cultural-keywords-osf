import argparse
from util import load_worldlex_data
import spacy
# spacy.require_gpu()

parser = argparse.ArgumentParser()

parser.add_argument("--spacy_model", default=None, type=str,
                    help="downloaded spacy model for lemmatization")
parser.add_argument("--lang", default=None, type=str,
                    help="language iso")
parser.add_argument("--output_file", default=None, type=str,
                    help="lemmatized output file")
parser.add_argument("--n_process", default=1, type=int,
                    help="batch size")
parser.add_argument("--bsz", default=None, type=int,
                    help="batch size")

args = parser.parse_args()

# worldlex frequency and fb dictionary
freq = load_worldlex_data(relative_frequency=False, average=False, languages=[args.lang])
words = list(freq[args.lang].keys())
print(f"loaded {len(words)} {args.lang} words")

nlp = spacy.load(args.spacy_model)
print(f"loaded {args.spacy_model} spacy model")

disable_pipes = ["parser",  "ner", "tok2vec"]
disable_pipes = [p for p in disable_pipes if p in nlp.pipe_names]

bsz = args.bsz if args.bsz is not None else len(words)

file = open(args.output_file, "w")
for word, doc in zip(words, nlp.pipe(words, disable=disable_pipes, n_process=args.n_process, batch_size=bsz)):
    for w in doc:
        file.write(",".join([word, w.lemma_]) + '\n')