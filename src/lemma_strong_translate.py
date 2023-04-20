from util import load_vectors_from_json, load_lemmatized, load_strong_translations, load_vectors
import os
import argparse
import spacy
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
#spacy.require_gpu()

parser = argparse.ArgumentParser()
parser.add_argument("--lemma_vectors_dir", default=None, type=str,
                    help="mean lemma vectors directory")
parser.add_argument("--aligned_vectors_dir", default=None, type=str, required=True,
                    help="fastText aligned vectors directory")
parser.add_argument("--lemma_dir", default=None, required=True,
                    help="words and lemma directory")
parser.add_argument("--strong_translations_dir", default=None, required=True,
                    help="strong translations directory")
parser.add_argument("--lang", default=None, type=str,
                    help="language iso")
parser.add_argument("--source_lang", default=None, type=str,
                    help="source language iso (en)")
parser.add_argument("--output_dir", default=None, type=str,
                    help="lemma strong translation output directory")
parser.add_argument("--source_spacy_model", default=None, type=str,
                    help="downloaded spacy model for lemmatization")
parser.add_argument("--sim_thresh", default=0, type=float,
                    help="vector similarity threshold")
args = parser.parse_args()

source_nlp = spacy.load(args.source_spacy_model)

disable_pipes = ["parser",  "ner", "tok2vec"]

for pipe in disable_pipes:
    if source_nlp and pipe in source_nlp.pipe_names:
        source_nlp.disable_pipes(pipe)

source_word_lemma_file = os.path.join(args.lemma_dir, "{}.txt".format(args.source_lang))

word_lemma_file = os.path.join(args.lemma_dir, "{}.txt".format(args.lang))


# word_vectors = load_vectors_from_json(args.lemma_vectors_dir,  languages=[args.source_lang, args.lang])
if args.lemma_vectors_dir is not None:
    mean_lemma_vectors = load_vectors_from_json(args.lemma_vectors_dir,  languages=[args.source_lang])[args.source_lang]

fpath = os.path.join(args.aligned_vectors_dir, "wiki.{}.align.vec")
lang_word_vectors = load_vectors(fpath.format(args.lang), None)
source_lang_word_vectors = load_vectors(fpath.format(args.source_lang), None)

word_lemma = {args.lang: load_lemmatized(word_lemma_file), args.source_lang: load_lemmatized(source_word_lemma_file)}

lemma_words = {lang: {} for lang in word_lemma}
for lang, wd_lm in word_lemma.items():
    for wd, lm in wd_lm.items():
        lemma_words[lang][lm] = lemma_words[lang].get(lm, [])
        lemma_words[lang][lm].append(wd)

strong_translations = load_strong_translations([args.lang], folder_path=args.strong_translations_dir,
                                               source=args.source_lang)

lemma_strong_translations = {"{}-{}_lemma".format(args.lang, args.source_lang):{},
                             "{}_lemma-{}".format(args.source_lang, args.lang): {}}

for lm, wds in lemma_words[args.lang].items():
    translated = {wd: strong_translations["{}-{}".format(args.lang, args.source_lang)][wd][0] # all translated forms within a lemma group
    if wd in strong_translations["{}-{}".format(args.lang, args.source_lang)] else None for wd in wds}

    l2s_lemmatized_translation = {} # convert translated form (English) to lemma form
    for w, translatedw in translated.items():
        if translatedw is None:
            l2s_lemmatized_translation[w] = None
        elif translatedw in word_lemma[args.source_lang]:
            l2s_lemmatized_translation[w] = word_lemma[args.source_lang][translatedw]
        else:
            l2s_lemmatized_translation[w]  = source_nlp(translatedw)[0].lemma_
    # where mean vector of lemma is taken
    if args.lemma_vectors_dir is not None:
        source_lemma_candidates = set([slemma for slemma in l2s_lemmatized_translation.values() if slemma is not None])
        source_lemma_vector = {slemma: np.expand_dims(mean_lemma_vectors[slemma], axis=0) for slemma in source_lemma_candidates
                               if slemma in mean_lemma_vectors}
        if source_lemma_vector:
            for w, lemmatizedw in l2s_lemmatized_translation.items():
                if lemmatizedw is None and w in lang_word_vectors:
                    w_vector = np.expand_dims(lang_word_vectors[w], axis=0)
                    cossim = {slemma: cosine_similarity(sl_vector, w_vector)[0] for slemma, sl_vector in
                              source_lemma_vector.items()}
                    max_lemma = max(cossim, key=cossim.get) if cossim else None
                    if max_lemma and cossim[max_lemma] >= args.sim_thresh:
                        l2s_lemmatized_translation[w] = max_lemma
    else: # take similarity between original words from strong-translations
        source_translation_candidates = set([translatedw for w, translatedw in translated.items() if translatedw is not None])
        source_translation_vector = {w: np.expand_dims(source_lang_word_vectors[w], axis=0) for w in source_translation_candidates
                                     if w in source_lang_word_vectors}
        if source_translation_vector:
            for w, lemmatizedw in l2s_lemmatized_translation.items():
                if lemmatizedw is None and w in lang_word_vectors:
                    w_vector = np.expand_dims(lang_word_vectors[w], axis=0)
                    cossim = {sword: cosine_similarity(svector, w_vector)[0] for sword, svector in
                              source_translation_vector.items()}
                    max_translation = max(cossim, key=cossim.get) if cossim else None
                    if max_translation and cossim[max_translation] >= args.sim_thresh:
                        if max_translation in word_lemma[args.source_lang]:
                            l2s_lemmatized_translation[w] = word_lemma[args.source_lang][max_translation]
                        else:
                            l2s_lemmatized_translation[w] = source_nlp(max_translation)[0].lemma_


    for w, lemmatizedw in l2s_lemmatized_translation.items():
        if lemmatizedw:
            lemma_strong_translations["{}-{}_lemma".format(args.lang, args.source_lang)][w] = lemmatizedw
            lemma_strong_translations["{}_lemma-{}".format(args.source_lang, args.lang)][lemmatizedw] = \
                    lemma_strong_translations["{}_lemma-{}".format(args.source_lang, args.lang)].get(lemmatizedw, [])
            lemma_strong_translations["{}_lemma-{}".format(args.source_lang, args.lang)][lemmatizedw].append(w)


source2langf = open(os.path.join(args.output_dir, "{}_lemma-{}.csv".format(args.source_lang, args.lang)), "w")
source2lang_rows = ["{},\"{}\"".format(slm, ",".join(word)) for slm, word in lemma_strong_translations["{}_lemma-{}".format(
    args.source_lang, args.lang)].items()]
source2langf.write("\n".join(source2lang_rows))

lang2sourcef = open(os.path.join(args.output_dir, "{}-{}_lemma.csv".format(args.lang, args.source_lang)), "w")
lang2source_rows = ["{},{}".format(word, slm) for word, slm  in lemma_strong_translations["{}-{}_lemma".format(
    args.lang, args.source_lang)].items()]
lang2sourcef.write("\n".join(lang2source_rows))

print("{} lemma translate complete.".format(args.lang))
