import csv
import os
import math
import copy
import json
import sys
import io
import numpy as np
from io import StringIO
from collections import Counter
import opencc
# converter = opencc.OpenCC('t2s.json')
converter = opencc.OpenCC('t2s')
csv.field_size_limit(sys.maxsize)

languages = ["ms", "pt", "fr", "id", "nl", "es", "en", "ru", "zh"]

# worldlex frequency
def load_worldlex_data(freq_dir="word-freq", relative_frequency=True, average=True, source=["blog", "twitter", "news"],
                       languages=['en', 'ms', 'pt', 'fr', 'id', 'nl', 'es', 'zh', 'ru']):

    file = {"en": ["Eng_US.Freq.2.txt"], "ms": ["My.Freq.2.txt"], "es": ["Es.Freq.2.txt", "Es_SA.Freq.2.txt"],
            "fr": ["Fre.Freq.2.txt"], "id": ["Id.Freq.2.txt"], "pt": ["PorEU.Freq.2.txt", "Por_Br.Freq.2.txt"],
            "nl": ["Nl.Freq.2.txt"], "ru": ["Ru.Freq.2.txt"], "zh": ["Chi.Freq.2.txt"], "ca": ["Cat.Freq.2.txt"], "da": ["DK.Freq.2.txt"],
            "fi": ["Fi.Freq.2.txt"], "de": ["De.Freq.2.txt"], "el": ["Gre.Freq.2.txt"], "it": ["Ita.Freq.2.txt"], "ko": ["Kr.Freq.2.txt"],
            "lt": ["Lit.Freq.2.txt"], "mk": ["Mk.Freq.2.txt"], "no": ["Nob.Freq.2.txt"], "pl": ["Pl.Freq.2.txt"], "ro": ["Ro.Freq.2.txt"],
            "sv": ["Swe.Freq.2.txt"], "uk": ["Uk.Freq.2.txt"]}

    data = {}

    for k, vs in file.items():
        if k in languages:
            file[k] = [os.path.join(freq_dir, v) for v in vs]
            data[k] = [[row.split('\t') for row in open(pt, 'r').read().rstrip().split('\n')] for pt in file[k] ]

    lang0 = languages[0]
    word_idx = data[lang0][0][0].index("Word")

    if relative_frequency:
        blog_idx = data[lang0][0][0].index("BlogFreqPm")
        tw_idx = data[lang0][0][0].index("TwitterFreqPm")
        news_idx = data[lang0][0][0].index("NewsFreqPm")
    else:
        blog_idx = data[lang0][0][0].index("BlogFreq")
        tw_idx = data[lang0][0][0].index("TwitterFreq")
        news_idx = data[lang0][0][0].index("NewsFreq")

    freq = {l: {} for l in languages}  # total frequency per million words
    for lang, dts in data.items():
        for dt in dts:
            for w in dt[1:]:
                try:
                    if w[word_idx] not in freq[lang]:
                        try:
                            freq[lang][w[word_idx]] = {"blog": float(w[blog_idx]), "twitter": float(w[tw_idx]),
                                                       "news": float(w[news_idx])}
                        except IndexError: # empty row
                            continue
                    else:
                        freq[lang][w[word_idx]]["blog"] += float(w[blog_idx])
                        freq[lang][w[word_idx]]["twitter"] += float(w[tw_idx])
                        freq[lang][w[word_idx]]["news"] += float(w[news_idx])
                except ValueError:
                    continue
    final_freq = {}
    for lang, w_dt in freq.items():
        final_freq[lang] = {}
        for w, fq in w_dt.items():
            final_freq[lang][w] = sum([freq[lang][w][src] for src in source])
            if average:
                final_freq[lang][w] = final_freq[lang][w] / len(source)
    return final_freq

# facebook dictionary
def load_fb_dictionaries():

    fb_bilingual_file = {"en-ms": "en-ms.txt", "ms-en": "ms-en.txt", "en-es": "en-es.txt", "es-en": "es-en.txt",
                         "en-fr": "en-fr.txt", "fr-en": "fr-en.txt", "en-pt": "en-pt.txt", "pt-en": "pt-en.txt",
                         "nl-en": "nl-en.txt", "en-nl": "en-nl.txt", "en-id": "en-id.txt", "id-en": "id-en.txt",
                         "zh-en": "zh-en.txt", "en-zh": "en-zh.txt", "en-ru": "en-ru.txt", "ru-en": "ru-en.txt"}

    fb_bilingual_data = {}
    for langs, f in fb_bilingual_file.items():
        fb_bilingual_file[langs] = os.path.join("bilingual-dict", f)
        if langs in ["en-ms", "ms-en", "en-pt", "pt-en", "en-nl", "nl-en", "en-id", "id-en"]:
            fb_bilingual_data[langs] = [row.split('\t') for row in open(fb_bilingual_file[langs], 'r').read().rstrip().split('\n')]
        else:
            fb_bilingual_data[langs] = [row.split(' ') for row in open(fb_bilingual_file[langs], 'r').read().rstrip().split('\n')]

    fb_bilingual_dict = {}  # e.g. {"ms-en": {'wilayah': ['province', 'region', 'territories', 'territory']}}
    for langs, dt in fb_bilingual_data.items():
        fb_bilingual_dict[langs] = {}
        for pair in dt:
            # multiple translations exist in multiple rows
            if pair[0] not in fb_bilingual_dict[langs]:
                fb_bilingual_dict[langs][pair[0]] = [pair[1]]
            else:
                fb_bilingual_dict[langs][pair[0]].append(pair[1])

    return fb_bilingual_dict


def load_strong_translations(target_languages, folder_path="strong-translations/freq-ratio-fasttext-wordnet", source="en"):
    strong_translations = {}

    for tar in target_languages:
        to_en = "{}-{}".format(tar, source)
        en_to = "{}-{}".format(source, tar)
        strong_translations[to_en] = {}
        strong_translations[en_to] = {}
        fpath = os.path.join(folder_path, "{}.csv".format(to_en))
        for trans in list(csv.reader(open(fpath, 'r'))):
            non_en_w = trans[0]
            en_w = trans[1]
            strong_translations[to_en][non_en_w] = [en_w]
            if en_w in strong_translations[en_to]:
                strong_translations[en_to][en_w] += [non_en_w]
            else:
                strong_translations[en_to][en_w] = [non_en_w]

    return strong_translations

# def save_dict_to_csv(data, csv_path):
#     with open(csv_path, 'w', encoding='UTF8') as f:
#         writer = csv.writer(f)
#         for k, v in data.items():
#             val = v if not isinstance(v, list) else " ".join(v)
#             writer.writerow([k, val])
#     return
#
# def load_dict_from_csv(csv_path):
#     data = {}
#     for row in list(csv.reader(open(csv_path, 'r'))):
#         data[row[0]] = row[1:]
#     return data

def load_vectors(fname, kwords=None):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        word = tokens[0].lower()
        emb = list(map(float, tokens[1:]))
        if kwords is None or word in kwords:
            if word in data:
                data[word] = np.concatenate((data[word], np.expand_dims(np.double(emb), axis=0)))
            else:
                data[word] = np.expand_dims(np.double(emb), axis=0)

    for word, dt in data.items():
        data[word] = np.mean(data[word], axis=0)

    return data

def load_vectors_from_json(foldername, languages=languages):
    vectors = {}

    for lang in languages:
        fpath = os.path.join(foldername, lang+".json")
        with open(fpath) as json_file:
            vectors[lang] = json.load(json_file)

    return vectors

def load_lemmatized(fpath):
    word_lemma = {}
    for row in open(fpath, "r").read().split('\n'):
        row = row.rstrip()
        srow = row.split(",")
        if row and '-' not in srow[0]:
            word_lemma[srow[0]] = srow[1]
    return word_lemma

def load_lemma2lemma_translations(languages, word_lemma, lemma_strong_translations, source="en_lemma",
                                  excluded_languages=["zh"]):
    lemma2lemma_translations = copy.deepcopy(lemma_strong_translations)

    for lang in languages:
        if lang not in excluded_languages:
            en_lemma_lemma = {}
            lemma_en_lemma = {}
            for en_lemma, trans in lemma_strong_translations[f"{source}-{lang}"].items():
                tar_lemmas = set([word_lemma[lang][tr].lower() for tr in trans])
                en_lemma_lemma[en_lemma] = list(tar_lemmas)
                for tar_lem in tar_lemmas:
                    lemma_en_lemma[tar_lem] = [en_lemma]
            lemma2lemma_translations[f"{source}-{lang}"] = en_lemma_lemma
            lemma2lemma_translations[f"{source}-en_lemma"] = lemma_en_lemma

    return lemma2lemma_translations


def load_swow_from_agg_data(fpath, delimiter="\t", cue_name="cue", response_name="response", fq_name="R1",
                            strength_name="R1.Strength", simplify_chinese=False):
    data = [list(csv.reader(StringIO(row), delimiter=delimiter, quoting=csv.QUOTE_NONE))[0]
            for row in open(fpath, "r").read().rstrip().split("\n")]
    column_name, agg_data = data[0], data[1:]
    cidx, ridx, fqidx, stridx = column_name.index(cue_name), column_name.index(response_name), column_name.index(
        fq_name), column_name.index(strength_name)

    r_fq, r_split_fq, cue_r_fq, r_cue_fq, cue_r_strength = {}, {}, {}, {}, {}

    for row in agg_data:
        cue, response, fq, strength = row[cidx], row[ridx], int(row[fqidx]), float(row[stridx])
        r_fq[response] = r_fq.get(response, 0) + fq
        for w in response.split(' '):
            r_split_fq[w] = r_split_fq.get(w, 0) + fq

        cue_r_fq[cue] = cue_r_fq.get(cue, {})
        cue_r_fq[cue][response] = cue_r_fq[cue].get(response, 0) + fq

        r_cue_fq[response] = r_cue_fq.get(response, {})
        r_cue_fq[response][cue] = r_cue_fq[response].get(cue, 0) + fq

        cue_r_strength[cue] = cue_r_strength.get(cue, {})
        cue_r_strength[cue][response] = {"weight": strength}

    r_tp = {r: len(cue_fq) for r, cue_fq in r_cue_fq.items()}

    return r_fq, r_split_fq, r_tp, cue_r_fq, r_cue_fq, cue_r_strength


def load_swow_responses(fpath, delimiter, cue_name="cue", r123_name=["R1", "R2", "R3"], ignored_responses=["NA"]):
    data = [list(csv.reader(StringIO(row), delimiter=delimiter))[0] for row in
            open(fpath, "r").read().rstrip().split("\n")]
    column_name, response = data[0], data[1:]
    cidx, r1idx, r2idx, r3idx = column_name.index(cue_name), column_name.index(r123_name[0]), column_name.index(
        r123_name[1]), column_name.index(r123_name[2])

    r1_fq, r123_fq = {}, {}
    r1_split_fq, r123_split_fq = {}, {}
    cue_r1_fq, cue_r123_fq = {}, {}
    r1_cue_fq, r123_cue_fq = {}, {}

    for row in response:
        cue, r1, r2, r3 = row[cidx], row[r1idx], row[r2idx], row[r3idx]

        r1_fq[r1] = r1_fq.get(r1, 0) + 1
        for splt_r1 in r1.split(" "):
            r1_split_fq[splt_r1] = r1_split_fq.get(splt_r1, 0) + 1

        cue_r1_fq[cue] = cue_r1_fq.get(cue, {})
        cue_r1_fq[cue][r1] = cue_r1_fq[cue].get(r1, 0) + 1

        r1_cue_fq[r1] = r1_cue_fq.get(r1, {})
        r1_cue_fq[r1][cue] = r1_cue_fq[r1].get(cue, 0) + 1

        r123_fq[r1] = r123_fq.get(r1, 0) + 1
        r123_fq[r2] = r123_fq.get(r2, 0) + 1
        r123_fq[r3] = r123_fq.get(r3, 0) + 1

        for response in [r1, r2, r3]:
            for splt_r in response.split(" "):
                r123_split_fq[splt_r] = r123_split_fq.get(splt_r, 0) + 1

        cue_r123_fq[cue] = cue_r123_fq.get(cue, {})
        cue_r123_fq[cue][r1] = cue_r123_fq[cue].get(r1, 0) + 1
        cue_r123_fq[cue][r2] = cue_r123_fq[cue].get(r2, 0) + 1
        cue_r123_fq[cue][r3] = cue_r123_fq[cue].get(r3, 0) + 1

        r123_cue_fq[r1], r123_cue_fq[r2], r123_cue_fq[r3] = r123_cue_fq.get(r1, {}), r123_cue_fq.get(r2,
                                                                                                     {}), r123_cue_fq.get(
            r3, {})
        r123_cue_fq[r1][cue] = r123_cue_fq[r1].get(cue, 0) + 1
        r123_cue_fq[r2][cue] = r123_cue_fq[r2].get(cue, 0) + 1
        r123_cue_fq[r3][cue] = r123_cue_fq[r3].get(cue, 0) + 1

    for ilval in ignored_responses:
        r1_fq.pop(ilval, None);
        r123_fq.pop(ilval, None)
        r1_cue_fq.pop(ilval, None);
        r123_cue_fq.pop(ilval, None)
        for cue in cue_r1_fq.keys():
            cue_r1_fq[cue].pop(ilval, None)
        for cue in cue_r123_fq.keys():
            cue_r123_fq[cue].pop(ilval, None)

    r1_tp, r123_tp = {r1: len(cue_fq.keys()) for r1, cue_fq in r1_cue_fq.items()}, {r: len(cue_fq.keys()) for r, cue_fq
                                                                                    in r123_cue_fq.items()}
    cue_r1_strength = {cue: {r1: {'weight': fq / sum(list(r1_fqs.values()))} for r1, fq in r1_fqs.items()} for
                       cue, r1_fqs in cue_r1_fq.items()}
    cue_r123_strength = {cue: {r: {'weight': fq / sum(list(r_fqs.values()))} for r, fq in r_fqs.items()} for cue, r_fqs
                         in cue_r123_fq.items()}

    return r1_fq, r123_fq, r1_split_fq, r123_split_fq, r1_tp, r123_tp, cue_r1_strength, cue_r123_strength, cue_r1_fq, cue_r123_fq


def expand_chinese_characters(bilingual_dict, convert="key", count_values=False):
    bi_dict = {}
    if convert == "key":
        for tchars, translations in bilingual_dict.items():
            schars = converter.convert(tchars)
            if count_values and isinstance(translations, dict):
                translations = Counter(translations)
            if schars not in bi_dict:
                bi_dict[schars] = translations
            else:
                bi_dict[schars] += translations
    else:
        if count_values:
            for key, tchars_stat in bilingual_dict.items():
                bi_dict[key] = {}
                for tchars, stat in tchars_stat.items():
                    schars = converter.convert(tchars)
                    if schars not in bi_dict[key]:
                        bi_dict[key][schars] = stat
                    else:
                        bi_dict[key][schars] += stat
        else:
            for source, tchar_list in bilingual_dict.items():
                bi_dict[source] = list(set([converter.convert(tchar) for tchar in tchar_list]))

    return bi_dict