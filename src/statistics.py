import math
import copy
import random
import numpy as np

smoothing_param = 20
random.seed(2022)


def rank_keyword_score(wordlist, languages, final_score, descending=True):  # (Eq. 1)
    # by default, higher scores ->  higher ranks
    rank = {l: {} for l in languages}  # {lang: {word: rank}}
    rank_keyword_score = {}  # {word (en): {"score": score, "lang": lang}}
    lang_word_norm_score = {l: {} for l in languages} # {lang: {word: score}}

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

        score_sum = sum(score.values())
        score_n = len(score.keys())
        for lang, sc in score.items():
            sc_avg_other = (score_sum - sc) / (score_n - 1)
            lang_word_norm_score[lang][w] = sc - sc_avg_other

        curr_keyword_score = sorted_score[0] - avg_other
        rank_keyword_score[w] = {"score": curr_keyword_score, "lang": sorted_langs_by_score[0]}

    return rank, rank_keyword_score, lang_word_norm_score


def get_n(freq):
    n = {lang: sum(dt.values()) for lang, dt in freq.items()}  # same across type of mappings
    return n


def get_alphas(freq, wordlist, smoothing_param):
    nm = len(set([w for wfq in freq.values() for w in wfq]))
    alpha_mi = smoothing_param
    alpha_0i = alpha_mi * nm
    alpha_m = {}
    alpha_0m = {}

    for w in wordlist:
        langs = [lang for lang, wfq in freq.items() if w in wfq.keys()]
        alpha_m[w] = len(langs) * alpha_mi
        alpha_0m[w] = len(langs) * alpha_mi * nm
    return (alpha_mi, alpha_0i, alpha_m, alpha_0m)


def final_score(freq, alphas, wordlist, lang_n, languages=None):
    languages = languages if languages is not None else freq.keys()

    def log_odds_ratio(mfq, n, alpha_m, alpha_0):
        return np.log((mfq + alpha_m) / (n + alpha_0 - mfq - alpha_m))

    def stddev(delta, wfq, prior_mfq, alpha_mi, alpha_m):
        return delta / ((1 / (wfq + alpha_mi)) + (1 / (prior_mfq + alpha_m))) ** 0.5

    alpha_mi, alpha_0i, alpha_m, alpha_0m = alphas

    m_n = {w: sum([lang_n[lang] for lang in languages if w in freq[lang].keys()]) for w in wordlist}
    m_fq = {w: sum([freq[lang].get(w, 0) for lang in languages]) for w in wordlist}
    m_lor = {w: log_odds_ratio(m_fq[w], m_n[w], alpha_m[w], alpha_0m[w]) for w in wordlist}

    m_fq_vect = np.array([m_fq[w] for w in wordlist])
    m_lor_vect = np.array([m_lor[w] for w in wordlist])
    alpha_m_vect = np.array([alpha_m[w] for w in wordlist])

    lang_word_score = {}
    for lang in languages:
        fq_vect = np.array([freq[lang].get(w, np.nan) for w in wordlist])
        lor_vect = log_odds_ratio(fq_vect, lang_n[lang], alpha_mi, alpha_0i)
        delta_vect = lor_vect - m_lor_vect
        delta_std_vect = stddev(delta_vect, fq_vect, m_fq_vect, alpha_mi, alpha_m_vect)

        lang_word_score[lang] = {w: val for w, val in dict(zip(wordlist, delta_std_vect)).items() if not np.isnan(val)}

    return lang_word_score


def proportion_score(freq, wordlist, lang_n):
    lang_word_score = {l: {} for l in freq}
    m_n = {w: sum([lang_n[lang] for lang, wfq in freq.items() if w in wfq.keys()]) for w in wordlist}
    m_fq = {w: sum([wfq.get(w, 0) for lang, wfq in freq.items()]) for w in wordlist}
    m_prop = {w: m_fq[w] / m_n[w] for w in wordlist}

    for lang in freq:
        other_langs = copy.deepcopy(list(freq.keys()))
        other_langs.remove(lang)
        for w in wordlist:
            if w in freq[lang]:
                prop_w = freq[lang][w] / lang_n[lang]
                prop_prior = m_prop[w]
                # avg_prop_others = sum(prop_others)/len(prop_others)
                lang_word_score[lang][w] = prop_w - prop_prior
    return lang_word_score

def frequency_score(wordlist, languages, freq, n):
    lang_word_score = {l: {} for l in languages}
    for lang in languages:
        for w in wordlist:
            if w in freq[lang]:
                lang_word_score[lang][w] = freq[lang][w] / n[lang]
    return lang_word_score


def sort_tuple_score(lang_word_score, descending=True):
    tuple_score = {}
    for lang, word_score in lang_word_score.items():
        for word, score in word_score.items():
            tuple_score[(lang, word)] = score
    sorted_tuples = sorted(tuple_score.keys(), key=tuple_score.get, reverse=True)
    return sorted_tuples


def get_recall_at_k(keywords, tuple_rank):
    recall_at_k = []
    num_recall = 0
    keyword_sz = sum([len(kws) for lang, kws in keywords.items()])
    for rank, (lang, word) in enumerate(tuple_rank):
        if word in keywords.get(lang, {}):
            num_recall += 1
        recall_at_k.append(num_recall / keyword_sz)
    return recall_at_k


def rank_keyword_by_freq_score(wordlist, languages, freq_score, descending=True):  # (Eq. 1)
    # language classification
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

