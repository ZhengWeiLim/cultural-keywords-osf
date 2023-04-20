import statistics
import numpy as np
import scipy.stats as stat
import random, math
from collections import Counter
random.seed(2022)
from scipy.stats import pearsonr, spearmanr
from src.statistics import get_alphas, get_n, rank_keyword_by_freq_score, random_baseline, proportion_score, \
    frequency_score, final_score, rank_keyword_score
from src.word_mapping import get_wordlist
from src.keywords import load_and_translate_keywords

def recall_rank_1_by_language(lang_keyword_dict, lang_word_rank_dict, language_names, show_found_keywords=True):
    lang_rank_1_kwords = {}
    recall = {}
    for lang, keywords in lang_keyword_dict.items():
        rank_1 = list(filter(lambda r: r[1] == 1, lang_word_rank_dict[lang].items()))
        rank_1_words = [r[0] for r in rank_1]
        total_rank_1 = len(rank_1)
        found_words = keywords.intersection(rank_1_words)
        lang_rank_1_kwords[lang] = found_words
        recall[lang] = len(found_words) / len(keywords) if keywords else "n/a"

        if show_found_keywords and language_names:
            print("{} words are highly associated with {}, of which {}/{} possible keywords are ranked 1" \
                  .format(total_rank_1, language_names[lang], len(found_words), len(keywords)))
            print(list(found_words))
            print("keywords that are not the highest ranked:")
            print(set(list(keywords)) - found_words)
            print()
    return recall, lang_rank_1_kwords


def mean_std_from_rank_by_language(lang_keyword_dict, lang_word_rank_dict):
    lang_mean = {}
    lang_std = {}
    for lang, kws in lang_keyword_dict.items():
        kword_ranks = [lang_word_rank_dict[lang][kw] for kw in kws if kw in lang_word_rank_dict[lang]]
        mean = sum(kword_ranks) / len(kword_ranks)
        if len(kword_ranks) > 1:
            std = statistics.stdev(kword_ranks)
        else:
            std = "n/a"
        lang_mean[lang] = mean
        lang_std[lang] = std
    return lang_mean, lang_std


def evaluate_by_keyword_categories_language(external_keywords, internal_keywords, available_keywords,
                                            language_names, lang_word_rank_dict, verbose=True, decimal_place=2):
    ext_recall, ext_r1_kwords = recall_rank_1_by_language(external_keywords, lang_word_rank_dict,
                                                          language_names, show_found_keywords=False)
    int_recall, int_r1_kwords = recall_rank_1_by_language(internal_keywords, lang_word_rank_dict,
                                                          language_names, show_found_keywords=False)
    all_recall, all_r1_kwords = recall_rank_1_by_language(available_keywords, lang_word_rank_dict,
                                                          language_names, show_found_keywords=False)

    ext_mean, ext_std = mean_std_from_rank_by_language(external_keywords, lang_word_rank_dict)
    int_mean, int_std = mean_std_from_rank_by_language(internal_keywords, lang_word_rank_dict)
    all_mean, all_std = mean_std_from_rank_by_language(available_keywords, lang_word_rank_dict)

    result = {"r1_recall": {"external": ext_recall, "internal": int_recall, "all": all_recall},
              "r1_keywords": {"external": ext_r1_kwords, "internal": int_r1_kwords, "all": all_r1_kwords},
              "mean": {"external": ext_mean, "internal": int_mean, "all": all_mean},
              "std": {"external": ext_std, "internal": int_std, "all": all_std}}

    if verbose:
        rd = decimal_place

        print("Recall of rank 1 keywords across languages: ")
        print("language, external, internal, overall")
        for lang, ext_r in ext_recall.items():
            ex_r = round(ext_r, rd) if ext_r != "n/a" else "n/a"
            in_r = round(int_recall[lang], rd) if int_recall[lang] != "n/a" else "n/a"
            all_r = round(all_recall[lang], rd) if all_recall[lang] != "n/a" else "n/a"
            print("{}: {}, {}, {}".format(lang, ex_r, in_r, all_r))

        print()
        print("(mean, std) of keyword ranks across languages: ")
        print("language, external, internal, overall")
        for lang, ext_m in ext_mean.items():
            ext_m = round(ext_m, rd) if ext_m != "n/a" else "n/a"
            ext_s = round(ext_std[lang], rd) if ext_std[lang] != "n/a" else "n/a"
            int_m = round(int_mean[lang], rd) if int_mean[lang] != "n/a" else "n/a"
            int_s = round(int_std[lang], rd) if int_std[lang] != "n/a" else "n/a"
            all_m = round(all_mean[lang], rd) if all_mean[lang] != "n/a" else "n/a"
            all_s = round(all_std[lang], rd) if all_std[lang] != "n/a" else "n/a"

            print("{}: ({},{}), ({},{}), ({},{})".format(lang, ext_m, ext_s, int_m, int_s, all_m, all_s))

    return result


def recall_mean_std_by_keyword_categories(external_keywords, internal_keywords, available_keywords, lang_word_rank_dict,
                                   excluded_language=None):
    internal_ranks = []
    external_ranks = []
    all_ranks = []
    for lang, kws in available_keywords.items():
        if excluded_language and lang == excluded_language:
            continue
        all_ranks += [lang_word_rank_dict[lang][kw] for kw in kws if kw in lang_word_rank_dict[lang]]
        external_ranks += [lang_word_rank_dict[lang][kw] for kw in external_keywords[lang] if kw in lang_word_rank_dict[lang]]
        internal_ranks += [lang_word_rank_dict[lang][kw] for kw in internal_keywords[lang] if kw in lang_word_rank_dict[lang]]

    int_mean = sum(internal_ranks) / len(internal_ranks)
    ext_mean = sum(external_ranks) / len(external_ranks)
    all_mean = sum(all_ranks) / len(all_ranks)

    int_std = statistics.stdev(internal_ranks)
    ext_std = statistics.stdev(external_ranks)
    all_std = statistics.stdev(all_ranks)

    int_recall = len(list(filter(lambda r: r==1, internal_ranks))) / len(internal_ranks)
    ext_recall = len(list(filter(lambda r: r == 1, external_ranks))) / len(external_ranks)
    all_recall = len(list(filter(lambda r: r == 1, all_ranks))) / len(all_ranks)

    result = {"mean": {"external": ext_mean, "internal": int_mean, "all": all_mean},
              "std": {"external": ext_std, "internal": int_std, "all": all_std},
              "recall": {"external": ext_recall, "internal": int_recall, "all": all_recall}}

    return result

def sort_rank_by_score(keyword_score_dict, reverse=True):
    sc = {w: s["score"] for w, s in  keyword_score_dict.items()}
    sorted_sc = sorted(sc, key=sc.get, reverse=reverse)
    word_sc_rank = {}
    for r, w in enumerate(sorted_sc):
        if r > 0:
            last_w = sorted_sc[r-1]
            if sc[last_w] == sc[w]:
                word_sc_rank[w] = word_sc_rank[last_w]
                continue
        word_sc_rank[w] = r+1

    return sc, sorted_sc, word_sc_rank


def get_list_of_count_by_range(arange, dlist):
    count = []
    dlist.sort()
    for i, x in enumerate(arange):
        if i < len(arange) - 1:
            group = list(filter(lambda d: d >= x and d < arange[i + 1], dlist))
        else:
            group = list(filter(lambda d: d >= x, dlist))
        dlist = [d for d in dlist if d not in group]
        count.append(len(group))
    return count


def get_recall_by_range_sum(external_keywords, internal_keywords, word_sc_rank_dict, lang_word_rank_dict, step):
    bins = range(0, len(list(word_sc_rank_dict.values())[0].keys()), step)

    def recall_by_range_sum(bins, word_ids, no_gold):
        count_list = get_list_of_count_by_range(bins, word_ids)
        recall_by_range = np.array(count_list) / no_gold
        sum_count_list = [count if i == 0 else count + sum(count_list[:i]) for i, count in enumerate(count_list)]
        recall_by_sum = np.array(sum_count_list) / no_gold
        return recall_by_range, recall_by_sum

    result = {}

    gold_cat_words = {}
    gold_cat_words["en-ext"] = list(external_keywords["en"])
    gold_cat_words["non-en-ext"] = [w for lang, wlist in external_keywords.items() for w in wlist if lang != "en"]
    gold_cat_words["en-int"] = list(internal_keywords["en"])
    gold_cat_words["non-en-int"] = [w for lang, wlist in internal_keywords.items() for w in wlist if lang != "en"]
    gold_cat_words["en-all"] = list(external_keywords["en"].union(internal_keywords["en"]))
    gold_cat_words["non-en-all"] = [w for lang, wlist in external_keywords.items()
                                    for w in wlist.union(internal_keywords[lang]) if lang != "en"]
    gold_cat_words["all-int"] = [w for wlist in internal_keywords.values() for w in wlist]
    gold_cat_words["all-ext"] = [w for wlist in external_keywords.values() for w in wlist]
    gold_cat_words["all"] = [w for lang, wlist in external_keywords.items()
                             for w in wlist.union(internal_keywords[lang])]

    for method, word_sc_rank in word_sc_rank_dict.items():
        non_en_ext_words = [w for lang, wlist in external_keywords.items()
                            for w in wlist if lang != "en" and lang_word_rank_dict[method][lang][w] == 1]
        non_en_int_words = [w for lang, wlist in internal_keywords.items()
                            for w in wlist if lang != "en" and lang_word_rank_dict[method][lang][w] == 1]
        non_en_all_words = [w for lang, wlist in external_keywords.items()
                            for w in wlist.union(internal_keywords[lang]) if lang != "en"
                            and lang_word_rank_dict[method][lang][w] == 1]
        en_ext_words = [w for w in external_keywords["en"] if lang_word_rank_dict[method]["en"][w] == 1]
        en_int_words = [w for w in internal_keywords["en"] if lang_word_rank_dict[method]["en"][w] == 1]
        en_all_words = [w for w in external_keywords["en"].union(internal_keywords["en"])
                        if lang_word_rank_dict[method]["en"][w] == 1]
        all_int_words = [w for lang, wlist in internal_keywords.items() for w in wlist
                         if lang_word_rank_dict[method][lang][w] == 1]
        all_ext_words = [w for lang, wlist in external_keywords.items() for w in wlist
                         if lang_word_rank_dict[method][lang][w] == 1]
        all_words = [w for lang, wlist in external_keywords.items()
                     for w in wlist.union(internal_keywords[lang]) if lang_word_rank_dict[method][lang][w] == 1]

        result[method] = {}

        non_en_ext_word_ids = [word_sc_rank[w] for w in non_en_ext_words]
        non_en_int_word_ids = [word_sc_rank[w] for w in non_en_int_words]
        non_en_all_word_ids = [word_sc_rank[w] for w in non_en_all_words]
        en_ext_word_ids = [word_sc_rank[w] for w in en_ext_words]
        en_int_word_ids = [word_sc_rank[w] for w in en_int_words]
        en_all_word_ids = [word_sc_rank[w] for w in en_all_words]
        all_int_word_ids = [word_sc_rank[w] for w in all_int_words]
        all_ext_word_ids = [word_sc_rank[w] for w in all_ext_words]
        all_word_ids = [word_sc_rank[w] for w in all_words]
        cat_word_ids = {"en-ext": en_ext_word_ids, "en-int": en_int_word_ids, "non-en-ext": non_en_ext_word_ids,
                        "non-en-int": non_en_int_word_ids, "en-all": en_all_word_ids,
                        "non-en-all": non_en_all_word_ids, "all-int": all_int_word_ids,
                        "all-ext": all_ext_word_ids, "all": all_word_ids}

        for cat, word_ids in cat_word_ids.items():
            result[method][cat] = {}
            result[method][cat]["range"], result[method][cat]["sum"] = recall_by_range_sum(bins, word_ids,
                                                                                           len(gold_cat_words[cat]))

    return result

def evaluate_keyword_classifications(classifier_lang_word_rank, external_keywords, internal_keywords, available_keywords):
    int_ext_recall_mean_std = {}

    max_padding = max([len(method) for method in classifier_lang_word_rank.keys()])
    print("method{}: recall, mean rank (ext/int/overall)".format(
        " "*(max_padding-len("method"))))

    for method, lang_word_rank in classifier_lang_word_rank.items():
        int_ext_recall_mean_std[method] = recall_mean_std_by_keyword_categories(external_keywords, internal_keywords,
                                                                 available_keywords, lang_word_rank,
                                                                 excluded_language=None)
        padding = " "*(max_padding - len(method))
        print("{}: ({:.2f}/{:.2f}/{:.2f}), ({:.2f}/{:.2f}/{:.2f})".format(method+padding,
            round(int_ext_recall_mean_std[method]["recall"]["external"],2),
            round(int_ext_recall_mean_std[method]["recall"]["internal"],2),
            round(int_ext_recall_mean_std[method]["recall"]["all"],2),
            round(int_ext_recall_mean_std[method]["mean"]["external"],2),
            round(int_ext_recall_mean_std[method]["mean"]["internal"],2),
            round(int_ext_recall_mean_std[method]["mean"]["all"],2)))


################################### CORRELATION ######################################


def pearson_correlation(word_score1, word_score2, label=None, verbose=True, outer=True):
    if outer:
        words = list(set(word_score1.keys()).union(set(word_score2.keys())))
    else:
        words = list(set(word_score1.keys()).intersection(set(word_score2.keys())))
    score1 = [word_score1[word] if word in word_score1 else 0 for word in words]
    score2 = [word_score2[word]  if word in word_score2 else 0 for word in words]
    corr, _ = pearsonr(score1, score2)
    if verbose:
        if label is None:
            print('Pearsons correlation: %.3f, number of words: {}'.format(len(words)) % corr)
        else:
            print('Pearsons correlation {}: %.3f, number of words: {}'.format(label, len(words)) % corr)
    return corr

def spearman_correlation(word_score1, word_score2, label=None, verbose=True, outer=True):
    if outer:
        words = list(set(word_score1.keys()).union(set(word_score2.keys())))
    else:
        words = list(set(word_score1.keys()).intersection(set(word_score2.keys())))
    score1 = [word_score1[word] if word in word_score1 else 0 for word in words]
    score2 = [word_score2[word]  if word in word_score2 else 0 for word in words]
    corr, _ = spearmanr(score1, score2)
    if verbose:
        if label is None:
            print('Spearman correlation: %.3f, number of words: {}'.format(len(words)) % corr)
        else:
            print('Spearman correlation {}: %.3f, number of words: {}'.format(label, len(words)) % corr)
    return corr



################################ INDEPENDENCE TESTS #################################

def sample_wordlist(wordlist, npartitions=100, drop_ratio=0.1, overlap=True, wreplacement=True):
    if overlap:
        wordlist = list(wordlist)
        nsamples = math.floor(len(wordlist) * (1-drop_ratio))
        if wreplacement:
            sampled_wordlists = [random.choices(wordlist, k=nsamples) for i in range(npartitions)]
        else:
            sampled_wordlists = [random.sample(wordlist, k=nsamples) for i in range(npartitions)]
    else:
        wordlist = list(wordlist)
        random.shuffle(wordlist)
        nwords = len(wordlist) // npartitions
        partitions = [wordlist[i:i+nwords] for i in range(0, nwords*npartitions, nwords)]
        sampled_wordlists = []
        for i in range(npartitions):
            curr_list = []
            for j, part in enumerate(partitions):
                if j!=i:
                    curr_list += part
            sampled_wordlists.append(curr_list)
    return sampled_wordlists

def expand_keywords(keywords, w_count):
    # expand keyword set based on repeated sampled words
    # register repeatedly sampled keywords as new keywords, filter keywords where count = 0
    new_keywords = {lang: set() for lang in keywords}
    for lang, kwords in keywords.items():
        for kword in kwords:
            kword_count = w_count.get(kword, 0)
            if kword_count > 1:
                for i in range(1, kword_count):
                    new_keywords[lang].add(f"{kword}_{i}")
            new_keywords[lang].add(kword)
    return new_keywords


def expand_tr_freq(tr_freq, wsamples):
    # register repeatedly sampled words as new words
    w_fq = {lang: {} for lang in tr_freq}
    w_count = {}
    for w in wsamples:
        for lang in tr_freq:
            if w in tr_freq[lang]:
                current_wcount = w_count.get(w, 0)
                if current_wcount > 0:
                    w_fq[lang][f"{w}_{current_wcount}"] = tr_freq[lang][w]
                else:
                    w_fq[lang][w] = tr_freq[lang][w]
        w_count[w] = w_count.get(w, 0) + 1
    return w_fq


def evaluate_by_vocab(wordlist, original_tr_freq, lemma_strong_translations, n, smoothing_param, recall_k=100, npartitions=10, keyword_languages=None):

    wordlist_clf_eval = []
    wordlist_rank_eval = []

    tr_freqs = []  # meaning class frequency
    w_counts = []  # meaning class counts (for resampled classes)

    languages = list(original_tr_freq.keys())
    keyword_languages = languages if keyword_languages is None else keyword_languages
    # bootstrap samples with replacements
    for wsamples in sample_wordlist(wordlist, npartitions, overlap=True, wreplacement=True, drop_ratio=0):
        w_fq = expand_tr_freq(original_tr_freq, wsamples)
        w_count = Counter(wsamples)
        tr_freqs.append(w_fq)
        w_counts.append(w_count)

    for tr_freq, w_count in zip(tr_freqs, w_counts):
        alpha_w, alpha = get_alphas(tr_freq, smoothing_param)
        wordlist = get_wordlist(tr_freq, languages)
        bayesian_score = final_score(wordlist, languages, tr_freq, alpha, alpha_w, n,
                                     smoothing_param=smoothing_param, stddev=True, log=True)
        # rank by s_w, word with S_w and argmax s_w
        bayesian_rank, bayesian_rank_keyword_score = rank_keyword_score(wordlist, languages, bayesian_score)

        # baseline method measures difference in proportions
        prop_score = proportion_score(wordlist, languages, tr_freq, n)
        prop_rank, prop_rank_keyword_score = rank_keyword_score(wordlist, languages, prop_score)

        # baseline method measures difference in raw frequency, rank by ascending and descending
        freq_score = frequency_score(wordlist, languages, tr_freq, n)
        asc_rank, asc_rank_keyword_score = rank_keyword_by_freq_score(wordlist, languages, freq_score, descending=False)
        des_rank, des_rank_keyword_score = rank_keyword_by_freq_score(wordlist, languages, freq_score, descending=True)
        random_rank = random_baseline(wordlist, tr_freq, languages)
        clf_ranks = {"bayesian": bayesian_rank, "proportion": prop_rank, "ascending": asc_rank, "descending": des_rank,
                     "random": random_rank}

        bayes_sc, bayes_sorted_sc, bayes_word_sc_rank = sort_rank_by_score(bayesian_rank_keyword_score)
        prop_sc, prop_sorted_sc, prop_word_sc_rank = sort_rank_by_score(prop_rank_keyword_score)
        asc_sc, asc_sorted_sc, asc_word_sc_rank = sort_rank_by_score(asc_rank_keyword_score, reverse=False)
        des_sc, des_sorted_sc, des_word_sc_rank = sort_rank_by_score(des_rank_keyword_score)

        word_sc_rank = {"bayesian": bayes_word_sc_rank, "proportion": prop_word_sc_rank, "ascending": asc_word_sc_rank,
                        "descending": des_word_sc_rank}

        cultural_keywords, concept_words_language, expected_keywords, all_keywords_en, available_keywords, \
        internal_keywords, external_keywords = load_and_translate_keywords(lemma_strong_translations, keyword_languages,
                                                                           "en_lemma", wordlist)

        internal_keywords = {lang: set(kws).intersection(set(tr_freq[lang].keys())) for lang, kws in
                             internal_keywords.items()}
        external_keywords = {lang: set(kws).intersection(set(tr_freq[lang].keys())) for lang, kws in
                             external_keywords.items()}
        available_keywords = {lang: set(kws).intersection(set(tr_freq[lang].keys())) for lang, kws in
                              available_keywords.items()}

        internal_keywords = expand_keywords(internal_keywords, w_count)
        external_keywords = expand_keywords(external_keywords, w_count)
        available_keywords = expand_keywords(available_keywords, w_count)

        int_ext_recall_mean_std = {}

        for method, lang_word_rank in clf_ranks.items():
            int_ext_recall_mean_std[method] = recall_mean_std_by_keyword_categories(external_keywords,
                                                                                    internal_keywords,
                                                                                    available_keywords, lang_word_rank,
                                                                                    excluded_language=None)

        wordlist_clf_eval.append(int_ext_recall_mean_std)
        wordlist_rank_eval.append(
            get_recall_by_range_sum(external_keywords, internal_keywords, word_sc_rank, clf_ranks, recall_k))
    return wordlist_clf_eval, wordlist_rank_eval



def get_result(method, metrics, wordlist_clf_eval, wordlist_rank_eval, rank_eval_idx=0):
    result = {}
    for metric in metrics:
        if "clf" == metric:
            result["clf-all-mean"]  = [clf_eval[method]["mean"]["all"] for clf_eval in wordlist_clf_eval]
            result["clf-all-recall"] = [clf_eval[method]["recall"]["all"] for clf_eval in wordlist_clf_eval]
        if "rank" == metric:
            result["rank-all-recall"] = [rank_eval[method]["all"]["sum"][rank_eval_idx] for rank_eval in wordlist_rank_eval]
    return result

def ttest_by_wordlist(method_pairs, wordlist_clf_eval, wordlist_rank_eval, metrics=["clf", "rank"]):

    pair_ttest = {}
    for tup in method_pairs:
        pair = "-".join(tup)
        pair_ttest[pair] = {}
        tup0_result = get_result(tup[0], metrics, wordlist_clf_eval, wordlist_rank_eval)
        tup1_result = get_result(tup[1], metrics, wordlist_clf_eval, wordlist_rank_eval)

        for metric, values0 in tup0_result.items():
            values1 = tup1_result[metric]
            pair_ttest[pair][metric] = stat.ttest_rel(values0, values1)
            pair_ttest[pair][f"{metric} mean"] = np.mean(np.array(values0) - np.array(values1))
            pair_ttest[pair][f"{metric} std"] = np.std(np.array(values0) - np.array(values1))

    return pair_ttest


def pvalue_by_wordlist(method_pairs, wordlist_clf_eval, wordlist_rank_eval, metrics=["clf", "rank"], rank_eval_idx=0):
    pair_pval = {}
    for tup in method_pairs:
        pair = "-".join(tup)
        pair_pval[pair] = {}
        tup0_result = get_result(tup[0], metrics, wordlist_clf_eval, wordlist_rank_eval, rank_eval_idx=rank_eval_idx)
        tup1_result = get_result(tup[1], metrics, wordlist_clf_eval, wordlist_rank_eval, rank_eval_idx=rank_eval_idx)

        for metric, values0 in tup0_result.items():
            values0 = np.array(values0)
            values1 = np.array(tup1_result[metric])
            pair_pval[pair][metric] = sum(values0 < values1) / len(wordlist_clf_eval)

    return pair_pval

def sample_mean_stddev(method, wordlist_clf_eval, wordlist_rank_eval, metrics=["clf", "rank"], rank_eval_idx=0):
    metric_res = get_result(method, metrics, wordlist_clf_eval, wordlist_rank_eval, rank_eval_idx=rank_eval_idx)
    metric_mean, metric_std = {}, {}
    for metric, res in metric_res.items():
        res = np.array(res)
        metric_mean[metric], metric_std[metric] = np.mean(res), np.std(res)
    return {"mean": metric_mean, "stddev": metric_std}