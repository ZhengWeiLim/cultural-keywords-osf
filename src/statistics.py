import math
import copy
import random

smoothing_param = 20
random.seed(2022)

def delta(fq_i, fq_j, a_w_i, a_w_j, a_i, a_j, n_i, n_j, log=True):
    if log:
        return math.log((fq_i + a_w_i) / (n_i + a_i - fq_i - a_w_i)) - math.log((fq_j + a_w_j) / (n_j + a_j - fq_j - a_w_j))
    else:
        return (fq_i + a_w_i) / (n_i + a_i - fq_i - a_w_i) - (fq_j + a_w_j) / (n_j + a_j - fq_j - a_w_j)


def alpha_w_other(lang, all_langs, freq, smoothing_param=smoothing_param):  # same across type of mappings
    other_langs = copy.deepcopy(all_langs)
    other_langs.remove(lang)
    other_uniq: int = sum([len(freq[tar].keys()) for tar in other_langs])
    return smoothing_param / other_uniq


def word_freq_J(word, lang, all_langs, freq):  # same across type of mappings
    other_langs = copy.deepcopy(all_langs)
    other_langs.remove(lang)
    other_fq = 0
    for oth in other_langs:
        if word in freq[oth]:
            other_fq += freq[oth][word]
    return other_fq


def n_other(lang, all_langs, n):  # same across type of mappings
    other_langs = copy.deepcopy(all_langs)
    other_langs.remove(lang)
    other_n = sum([n[tar] for tar in other_langs])
    return other_n


def variance_delta(word, lang, all_langs, alpha, freq, smoothing_param=smoothing_param):
    if word in freq[lang]:
        fq_lang = freq[lang][word]
        fq_J = word_freq_J(word, lang, all_langs, freq)
        alpha_lang = alpha[lang]
        alpha_J = alpha_w_other(lang, all_langs, freq, smoothing_param=smoothing_param)
        return (1 / (fq_lang + alpha_lang)) + (1 / (fq_J + alpha_J))
    else:
        raise ValueError('{} is not found in {}'.format(word, lang))


def final_score(wordlist, languages, freq, alpha, alpha_w, n, smoothing_param=smoothing_param, stddev=True, log=True):
    lang_delta = {}  # {lang: {w: delta}}

    for lang in languages:
        delt = {}
        for w in wordlist:
            if w in freq[lang]:
                delt[w] = delta(freq[lang][w], word_freq_J(w, lang, languages, freq),
                                alpha_w[lang], alpha_w_other(lang, languages, freq, smoothing_param=smoothing_param),
                                smoothing_param, smoothing_param, n[lang], n_other(lang, languages, n), log=log)
        lang_delta[lang] = delt

    if stddev:
        lang_final_score = {}

        for lang, delt in lang_delta.items():

            lang_final_score[lang] = {}

            for w, d in delt.items():
                var_wd = variance_delta(w, lang, languages, alpha, freq, smoothing_param=smoothing_param)
                lang_final_score[lang][w] = d / (var_wd ** 0.5)
    else:
        lang_final_score = lang_delta

    return lang_final_score  # {lang: {word: score}}


def rank_keyword_score(wordlist, languages, final_score, descending=True):  # (Eq. 1)
    # by default, higher scores ->  higher ranks
    rank = {l: {} for l in languages}  # {lang: {word: rank}}
    rank_keyword_score = {}  # {word (en): {"score": score, "lang": lang}}

    for w in wordlist:
        score = {}
        for lang in languages:
            if w in final_score[lang]:
                score[lang] = final_score[lang][w]
        sorted_langs_by_score = sorted(score, key=score.get, reverse=descending)
        for i, lang in enumerate(sorted_langs_by_score):
            if i > 0 and score[lang] == score[sorted_langs_by_score[i-1]]:
                rank[lang][w] = rank[sorted_langs_by_score[i-1]][w]
            else:
                rank[lang][w] = i + 1

        sorted_score = sorted(score.values(), reverse=descending)
        avg_other = sum(sorted_score[1:]) / len(sorted_score[1:])
        curr_keyword_score = sorted_score[0] - avg_other
        rank_keyword_score[w] = {"score": curr_keyword_score, "lang": sorted_langs_by_score[0]}

    return rank, rank_keyword_score

def get_alphas(freq, smoothing_param=smoothing_param):
    alpha_w = {lang: smoothing_param / len(list(word_fq.keys())) for lang, word_fq in freq.items()}
    alpha = {lang: smoothing_param for lang, dt in freq.items()}  # same across type of mappings
    return alpha_w, alpha

def get_n(freq):
    n = {lang: sum(dt.values()) for lang, dt in freq.items()}  # same across type of mappings
    return n

def proportion_score(wordlist, languages, freq, n):
    lang_word_score = {l: {} for l in languages}
    for lang in languages:
        other_langs = copy.deepcopy(languages)
        other_langs.remove(lang)
        for w in wordlist:
            if w in freq[lang]:
                prop_w = freq[lang][w] / n[lang]
                prop_others = [freq[othl][w] / n[othl] for othl in other_langs if w in freq[othl]]
                avg_prop_others = sum(prop_others)/len(prop_others)
                lang_word_score[lang][w] = prop_w - avg_prop_others
    return lang_word_score

def frequency_score(wordlist, languages, freq, n):
    lang_word_score = {l: {} for l in languages}
    for lang in languages:
        for w in wordlist:
            if w in freq[lang]:
                lang_word_score[lang][w] = freq[lang][w] / n[lang]
    return lang_word_score

def rank_keyword_by_freq_score(wordlist, languages, freq_score, descending=True):  # (Eq. 1)
    # by default, higher scores ->  higher ranks
    rank = {l: {} for l in languages}  # {lang: {word: rank}}
    rank_keyword_score = {}  # {word (en): {"score": score, "lang": lang}}

    for w in wordlist:
        score = {}
        for lang in languages:
            if w in freq_score[lang]:
                score[lang] = freq_score[lang][w]
        sorted_langs_by_score = sorted(score, key=score.get, reverse=descending)
        for i, lang in enumerate(sorted_langs_by_score):
            rank[lang][w] = i + 1
        curr_keyword_score = sum(score.values())/len(score.values())
        rank_keyword_score[w] = {"score": curr_keyword_score, "lang": sorted_langs_by_score[0]}

    return rank, rank_keyword_score

##################### OTHER BASELINES FOR LANGUAGE CLASSIFICATIONS ###################

# rank languages in random
def random_baseline(wordlist, freq, languages):
    lang_word_rank = {l: {} for l in languages}
    for w in wordlist:
        w_langs = [lang for lang in languages if w in freq[lang]]
        random.shuffle(w_langs)
        for rank, lang in enumerate(w_langs):
            lang_word_rank[lang][w] = rank+1
    return lang_word_rank


# rank english as 1 if the word exists in english corpus, rank randomly otherwise
def always_english_baseline(wordlist, freq, languages):
    lang_word_rank = {l: {} for l in languages}
    for w in wordlist:
        w_langs = [lang for lang in languages if w in freq[lang]]
        random.shuffle(w_langs)
        if "en" in w_langs:
            w_langs.remove("en")
            lang_word_rank["en"][w] = 1
            for rank, lang in enumerate(w_langs):
                lang_word_rank[lang][w] = rank + 2
        else:
            for rank, lang in enumerate(w_langs):
                lang_word_rank[lang][w] = rank + 1
    return lang_word_rank


# rank the majority non-english language as 1 if word exists in the language, otherwise, rank randomly
def majority_baseline_non_english(wordlist, freq, languages, gold_keywords):
    lang_kword_sz = {lang: len(gold_keywords.get(lang, [])) for lang in languages}
    lang_kword_sz.pop("en", None)
    majority_langs = sorted(lang_kword_sz, key=lang_kword_sz.get, reverse=True)
    lang_word_rank = {l: {} for l in languages}
    for w in wordlist:
        w_langs = [lang for lang in majority_langs if w in freq[lang]]
        majority = w_langs[0]
        w_langs.remove(majority)
        if w in freq["en"]:
            w_langs.append("en")
        random.shuffle(w_langs)
        lang_word_rank[majority][w] = 1
        for rank, lang in enumerate(w_langs):
            lang_word_rank[lang][w] = rank + 2

    return lang_word_rank

