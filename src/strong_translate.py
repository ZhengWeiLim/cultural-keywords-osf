import argparse
import os
import opencc
converter = opencc.OpenCC('t2s.json')
import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet as wn

from util import load_worldlex_data, load_fb_dictionaries, expand_chinese_characters, load_vectors
from src.word_mapping import get_strong_translations_through_ratio_similarity_score

parser = argparse.ArgumentParser()
parser.add_argument("--lang", default=None, type=str,
                    help="mean lemma vectors directory")
parser.add_argument("--fasttext_emb_dir", default=None, type=str, required=True,
                    help="fastText aligned vectors directory")
parser.add_argument("--freq_dir", default=None, type=str, required=True,
                    help="worldlex frequency directory")
parser.add_argument("--output_dir", default=None, type=str,
                    help="strong translation output directory")
args = parser.parse_args()

source, target = "en", args.lang
languages = [source, target]
freq = load_worldlex_data(freq_dir=args.freq_dir, relative_frequency=False, average=False, languages=languages)

fb_bilingual_file= {f'en-{target}': f'en-{target}.txt', f'{target}-en': f'{target}-en.txt'}
fb_bilingual_dict = load_fb_dictionaries(fb_bilingual_file)

if target == "zh":
    fb_bilingual_dict["zh-en"] = expand_chinese_characters(fb_bilingual_dict["zh-en"], convert="key")
    fb_bilingual_dict["en-zh"] = expand_chinese_characters(fb_bilingual_dict["en-zh"] , convert="value")

vectors = {}
fasttext_emb_file = "wiki.{}.align.vec"

for lang in languages:
    fname = fasttext_emb_file.format(lang)
    fpath = os.path.join(args.fasttext_emb_dir, fname)
    all_words = set(freq[lang].keys())
    if lang != "en":
        all_words.update(set(fb_bilingual_dict["{}-en".format(lang)].keys()))
        all_words.update(set([tr for trans in fb_bilingual_dict["en-{}".format(lang)].values() for tr in trans]))
    else:
        for l in languages:
            if l == "en":
                continue
            all_words.update(set([tr for trans in fb_bilingual_dict["{}-en".format(l)].values() for tr in trans]))
            all_words.update(set(fb_bilingual_dict["en-{}".format(l)].keys()))
    vectors[lang] = load_vectors(fpath, all_words)

strong_translations = get_strong_translations_through_ratio_similarity_score([target], source, freq, fb_bilingual_dict,
                                                           vectors, min_fq=0.1, min_sim=0.001, fq_thresh=0.75,
                                                           lo_sim_thresh=0.2, hi_sim_thresh=0.3, wordnet=wn)

for pair, trans in strong_translations.items():
    langf = open(f"{args.output_dir}/{pair}.csv", "w")
    trans = [src + ',\"'+ ",".join(tar) + '\"' for src, tar in trans.items()]
    langf.write('\n'.join(trans))