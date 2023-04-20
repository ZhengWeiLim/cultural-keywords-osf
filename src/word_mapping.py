import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import wordnet as wn


def get_strong_translations_through_ratio_similarity_score(target_languages, source_language, freq, bilingual_dict,
                                                           vectors, min_fq=0.1, min_sim=0.001, fq_thresh=0.75,
                                                           lo_sim_thresh=0.2, hi_sim_thresh=0.3, wordnet=wn):
    strong_translations = {}  # pairs of translation that pass the strong translations_rules

    def get_trans_freq_ratio(w, tar, source_language, bilingual_dict, freq, vectors, min_fq, wordnet=None):
        if w in bilingual_dict[tar + "-" + source_language]:
            cand_trans = bilingual_dict[tar + "-" + source_language][w]  # candidate translations
            trans_freq = {}
            for tr in cand_trans:
                # exclude translation to english that is not an english word
                if (wordnet and len(wordnet.synsets(tr)) == 0) and source_language == "en":
                    continue
                if tr in freq[source_language]:
                    trans_freq[tr] = freq[source_language][tr]['avg']
                else:
                    trans_freq[tr] = 0
            sum_trans_freq = max(sum(trans_freq.values()), min_fq)
            trans_freq_ratio = {tr: fq / sum_trans_freq for tr, fq in trans_freq.items()}

            return trans_freq_ratio
        return None

    for tar in target_languages:

        strong_translations[tar + "-" + source_language] = {}
        strong_translations[source_language + "-" + tar] = {}

        for w, dt in freq[tar].items():
            # does not include non-english words, include english words that do not exist in the english corpus.
            tr_sc = get_trans_freq_ratio(w, tar, source_language, bilingual_dict, freq, vectors, min_fq=min_fq,
                                         wordnet=wordnet)
            best_tr = None
            if tr_sc:  # not None and empty
                w_sc = {}
                trans_sim = {}

                w_vect = np.expand_dims(vectors[tar][w], axis=0) if w in vectors[tar] else None

                for tr in tr_sc.keys():
                    tr_tr_sc = get_trans_freq_ratio(tr, source_language, tar, bilingual_dict, freq, vectors,
                                                    min_fq=min_fq, wordnet=None)
                    if tr_tr_sc is not None and w in tr_tr_sc:
                        w_sc[tr] = tr_tr_sc[w]
                    else:
                        w_sc[tr] = 0

                    if w_vect is not None and tr in vectors[source_language]:
                        tr_vect = np.expand_dims(vectors[source_language][tr], axis=0)
                        trans_sim[tr] = cosine_similarity(tr_vect, w_vect)[0]
                    else:
                        trans_sim[tr] = min_sim

                # above fq_thresh ratio in both direction
                above_fq_threshold = list(filter(lambda tr: tr_sc[tr] >= fq_thresh and w_sc[tr] >= fq_thresh
                                                 , tr_sc.keys()))
                # and pass similarity test >= lo_thresh
                if above_fq_threshold and trans_sim[above_fq_threshold[0]] >= lo_sim_thresh:
                    best_tr = above_fq_threshold[0]
                else:
                    # or the one with highest similarity score that passes >= hi_thresh
                    sorted_tr = sorted(trans_sim, key=trans_sim.get, reverse=True)
                    best_tr_sim = sorted_tr[0]
                    best_tr_sim_sc = trans_sim[best_tr_sim]
                    if best_tr_sim_sc >= hi_sim_thresh:
                        best_tr = best_tr_sim

                if best_tr is not None:
                    if w in strong_translations[tar + "-" + source_language]:
                        strong_translations[tar + "-" + source_language][w] += [best_tr]
                    else:
                        strong_translations[tar + "-" + source_language][w] = [best_tr]
                    if best_tr in strong_translations[source_language + "-" + tar]:
                        strong_translations[source_language + "-" + tar][best_tr] += [w]
                    else:
                        strong_translations[source_language + "-" + tar][best_tr] = [w]

    return strong_translations


def calculate_frequency(freq, strong_translations, languages, source):
    pro_freq = {}

    for lang in languages:
        pro_freq[lang] = {}

        if lang == source:
            pro_freq[lang] = {w: dt for w, dt in freq[source].items()}
        else:
            for w, fq in freq[lang].items():
                if w in strong_translations[lang + "-" + source]:
                    if isinstance(strong_translations[lang + "-" + source][w], list):
                        source_tr = strong_translations[lang + "-" + source][w][0]  # in english
                    else:
                        source_tr = strong_translations[lang + "-" + source][w]
                    if source_tr in pro_freq[lang]:
                        pro_freq[lang][source_tr] += fq
                    else:
                        pro_freq[lang][source_tr] = fq
    return pro_freq


def get_wordlist(freq, languages):
    # combine wordlist across languages, word must exist in more than 1 language
    word_no = {}
    for lang in languages:
        for w in freq[lang].keys():
            if w in word_no:
                word_no[w] += 1
            else:
                word_no[w] = 1
    wordlist = [wn[0] for wn in filter(lambda wn: wn[1] > 1, word_no.items())]
    return wordlist
