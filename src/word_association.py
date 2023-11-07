from src.word_mapping import calculate_frequency, get_wordlist
from src.statistics import get_alphas, get_n, final_score, rank_keyword_score
from src.evaluation import sort_rank_by_score
import copy
import matplotlib.pyplot as plt


def preprocess_fq(lang_word_fq, column_name="avg"):
    return {lang: {word: {column_name: fq} for word, fq in word_fq.items()} for lang, word_fq in lang_word_fq.items()}

def flatten_fq(lang_word_fq, column_name="avg"):
    return {lang: {word: fq[column_name]for word, fq in word_fq.items()} for lang, word_fq in lang_word_fq.items()}

def get_wordlist_scores(w_freq, smoothing_param, languages, strong_translations, translate=False, source="en_lemma", ):
    if translate:
        w_freq = calculate_frequency(w_freq, strong_translations, languages, source)
    mwordlist = get_wordlist(w_freq, languages)
    alphas_  = get_alphas(w_freq, mwordlist, smoothing_param)
    n_ = get_n(w_freq)
    # print(n_, alphas_)
    mclf_lang_word_score = final_score(w_freq, alphas_, mwordlist,  n_, languages=languages)

    mclf_ranks, mclf_rank_keyword_score, _ = rank_keyword_score(mwordlist, languages, mclf_lang_word_score)
    msc, msorted_sc, mword_sc_rank = sort_rank_by_score(mclf_rank_keyword_score)
    return w_freq, mwordlist, mclf_lang_word_score, mclf_ranks, mclf_rank_keyword_score, msc, msorted_sc, mword_sc_rank



def translate_cue_response_stat(cue_response_stat, languages, strong_translations, source, translate="cue",
                                non_translatable_key=None):
    # translate = "cue" or "response"
    # all non translatable words will be collected in non_translatable_key if it is not None
    if translate == "cue":
        cuetr_response_stat = {}
        for lang in languages:
            cuetr_response_stat[lang] = {}
            if lang == "en":
                cuetr_response_stat[lang] = cue_response_stat[lang]
            else:
                for cue, response_stat in cue_response_stat[lang].items():
                    if cue in strong_translations["{}-{}".format(lang, source)]:
                        cuetr = strong_translations["{}-{}".format(lang, source)][cue][0]
                        cuetr_response_stat[lang][cuetr] = response_stat
                    elif non_translatable_key is not None:
                        cuetr_response_stat[lang][non_translatable_key] = cuetr_response_stat[lang].get(
                            non_translatable_key, {})
                        cuetr_response_stat[lang][non_translatable_key].update(response_stat)

        return cuetr_response_stat

    elif translate == "response":
        cue_responsetr_stat = {}
        for lang in languages:
            cue_responsetr_stat[lang] = {}
            if lang == "en":
                cue_responsetr_stat[lang] = cue_response_stat[lang]
            else:
                for cue, response_stat in cue_response_stat[lang].items():
                    cue_responsetr_stat[lang][cue] = {}
                    for response, stat in response_stat.items():
                        if response in strong_translations["{}-{}".format(lang, source)]:
                            responsetr = strong_translations["{}-{}".format(lang, source)][response][0]
                            cue_responsetr_stat[lang][cue][responsetr] = stat
                        elif non_translatable_key is not None:
                            cue_responsetr_stat[lang][cue][non_translatable_key] = cue_responsetr_stat[lang][cue].get(
                                non_translatable_key, 0) + stat

        return cue_responsetr_stat


def get_response_sum_with_cues(cue_r_fq, cues=None, initial_r_fq=None):
    r_fq = {}
    for cue, rfq in cue_r_fq.items():
        if (cues is None) or (cue in cues):
            for r, fq in rfq.items():
                r_fq[r] = r_fq.get(r, 0) + fq
        else:
            for r, fq in rfq.items():
                r_fq[r] = r_fq.get(r, 0) + 0
    if initial_r_fq is not None:
        r_fq = {r: r_fq[r] if r in r_fq else 0 for r, _ in initial_r_fq.items()}
    return r_fq


def get_rescaled_response_distribution_with_uniform_cues(tr_cues_tr_response_stat, shared_vocabulary,
                                                         cue_fq_threshold=0, verbose=False):
    languages = list(tr_cues_tr_response_stat.keys())
    trcues_trresponse_stat = copy.deepcopy(tr_cues_tr_response_stat)

    # deleting out of shared_vocabulary entries
    for lang, trc_trr_stat in trcues_trresponse_stat.items():
        oov_cues = set(trc_trr_stat.keys()) - set(shared_vocabulary)
        for cue in oov_cues:
            trcues_trresponse_stat[lang].pop(cue, None)

    for lang, trc_trr_stat in trcues_trresponse_stat.items():
        for cue in trc_trr_stat.keys():
            oov_responses = set(trc_trr_stat[cue].keys()) - set(shared_vocabulary)
            for response in oov_responses:
                trcues_trresponse_stat[lang][cue].pop(response, None)

    # retain only cue-responses where total responses per cue >= cue_fq_threshold
    if cue_fq_threshold > 0:
        if verbose:
            print("Maximum and minimum cue frequencies")
            for lang in languages:
                print("{}, max: {}, min: {}".format(lang, max([sum(rfq.values()) for cue, rfq in
                                                               trcues_trresponse_stat[lang].items()]),
                                                    min([sum(rfq.values()) for cue, rfq in
                                                         trcues_trresponse_stat[lang].items()])))
            plt.hist([[sum(rfq.values()) for cue, rfq in trcues_trresponse_stat[lang].items()] for lang in languages],
                     label=languages, density=False, bins=range(10, 310, 10))
            plt.legend(fontsize=12)
            plt.title("Cue frequency distribution after controlling vocabulary")
            plt.show()
        trcues_trresponse_stat = {lang: {cue: rfq for cue, rfq in trcues_trresponse_stat[lang].items() if
                                         sum(rfq.values()) >= cue_fq_threshold} for lang in languages}

    # cues shared by all languages
    shared_cues = trcues_trresponse_stat["en"].keys()
    for lang in trcues_trresponse_stat:
        shared_cues = set(shared_cues).intersection(set(trcues_trresponse_stat[lang].keys()))

    if verbose:
        print("Number of shared_cues: {}".format(len(shared_cues)))

    # keep entries belong in shared_cues
    trcues_trresponse_stat = {
        lang: {cue: rfq for cue, rfq in trcues_trresponse_stat[lang].items() if cue in shared_cues} for lang in
        languages}

    # scaling to minimum size of cues for every language
    cue_stat = {lang: {cue: sum(rstat.values()) for cue, rstat in cue_rstat.items()} for lang, cue_rstat in
                trcues_trresponse_stat.items()}
    min_cue_stat = min([csum for lang, cuesum in cue_stat.items() for csum in cuesum.values()])
    trcues_trresponse_stat = {
        lang: {cue: {r: stat / cue_stat[lang][cue] * min_cue_stat for r, stat in rstat.items()}  # diff number of cues
               for cue, rstat in cue_rstat.items()} for lang, cue_rstat in trcues_trresponse_stat.items()}
    cue_stat = {lang: {cue: sum(rstat.values()) for cue, rstat in cue_rstat.items()} for lang, cue_rstat in
                trcues_trresponse_stat.items()}
    if verbose:
        # cue frequency
        print("Number of cues after filtering")
        print({lang: len(cuestat) for lang, cuestat in cue_stat.items()})

    # collecting response frequency over rescaled data
    response_stat_sum = {lang: sum([stat for rstat in cue_rstat.values() for stat in rstat.values()]) for
                         lang, cue_rstat in trcues_trresponse_stat.items()}
    min_sum = min(response_stat_sum.values())
    response_stat_sum = {lang: min_sum for lang in response_stat_sum}

    # uniform cue weight and redistribute responses based on cue weight
    cue_weight = {lang: {cue: 1 / len(cue_stat[lang]) for cue in cue_stat[lang]} for lang in cue_stat}

    cue_nresponse = {lang: {cue: rstatsum * cue_weight[lang][cue] for cue in cue_stat[lang]} for lang, rstatsum in
                     response_stat_sum.items()}
    if verbose:
        print({lang: sum(cueweight.values()) for lang, cueweight in cue_nresponse.items()})

    scaled_trcues_trresponse_stat = {lang: {} for lang in trcues_trresponse_stat}

    for lang, cue_rstat in trcues_trresponse_stat.items():
        for cue, rstat in cue_rstat.items():
            sum_rstat = sum(rstat.values())
            scaled_trcues_trresponse_stat[lang][cue] = {r: (stat / sum_rstat) * cue_nresponse[lang][cue] for r, stat in
                                                        rstat.items()}

    trcues_trresponse_stat = scaled_trcues_trresponse_stat

    # overall response distribution (without cues), in frequency or sum of strength if normalized
    cr1ccue_sum_of_factor = {lang: get_response_sum_with_cues(trc_trr_stat) for lang, trc_trr_stat in
                             trcues_trresponse_stat.items()}

    # set any of shared vocabulary that does not exist in a language to be 0
    for lang, cr1_strength in cr1ccue_sum_of_factor.items():
        leftover_responses = set(shared_vocabulary) - set(cr1_strength.keys())
        for r in leftover_responses:
            cr1ccue_sum_of_factor[lang][r] = 0

    if verbose:
        plt.hist([[sum(rfq.values()) for cue, rfq in trcues_trresponse_stat[lang].items()] for lang in languages],
                 label=languages, density=False, bins=range(10, 310, 10))
        plt.legend(fontsize=12)
        plt.title("Cue frequency distribution after filtering by threshold and scaling")
        plt.show()

        plt.hist([[fq for fq in cr1ccue_sum_of_factor[lang].values()] for lang in languages], label=languages,
                 density=False, bins=range(10, 310, 10))
        plt.legend(fontsize=12)
        plt.title("Response frequency distribution after filtering by threshold and scaling")
        plt.show()

    return cr1ccue_sum_of_factor, trcues_trresponse_stat, shared_cues

def get_lemma_set(language, word, word_lemma, strong_translations, freq):
    if language != "zh":
        lemma_set = {}
        for msword in strong_translations["en_lemma-{}".format(language)][word]:
            lemma_set[word_lemma[language][msword]] = lemma_set.get(word_lemma[language][msword], []) + [msword]

        lemma_set = {lem if lem in words else max(words, key =lambda w: freq[language][w])  for lem, words in lemma_set.items()}
        code_switch = list(sorted(filter(lambda w: w in strong_translations["en-en_lemma"], lemma_set), key=lambda w: freq[language][w], reverse=True))
        not_code_switch = list(sorted([lem for lem in lemma_set if lem not in code_switch], key=lambda w: freq[language][w], reverse=True))
        lemma_set = not_code_switch + code_switch
    return lemma_set


def get_usage_assoc_alignment(usage_label, assoc_label, languages, language_names, strong_translations, shared_vocabulary,
                              clf_rank_keyword_score, clf_tuple, sc, word_lemma, freq, latex_format=False, available_keywords=None):
    correl = {"nkeywords": {}, "nclassified": {}}

    for metric in correl:
        for method in [usage_label]:
            correl[metric][method] = {"r123": {lang: {} for lang in languages}}

    topswowr123b, topswowr123p, topswowr123des = {}, {}, {}
    topfreqb, topfreqp, topfreqdes = {}, {}, {}
    # swowr123_freq_b, swowr123_freq_p, swowr123_freq_des = {}, {}, {}
    # swowr123_b, freqr123_b, swowr123_p, freqr123_p, swowr123_des, freqr123_des = {}, {}, {}, {}, {}, {}
    topswowr123b_tuple, topfreqb_tuple = {}, {}

    for lang in languages:
        topswowr123b[lang] = sorted([w for w in shared_vocabulary
                                     if clf_rank_keyword_score[assoc_label][w]["lang"] == lang],
                                    key=sc[assoc_label][lang].get,
                                    reverse=True)

        topfreqb[lang] = sorted([w for w in shared_vocabulary
                                 if clf_rank_keyword_score[usage_label][w]["lang"] == lang], key=sc[usage_label][lang].get,
                                reverse=True)
        topswowr123b_tuple[lang] = [w for lang2, w in clf_tuple[assoc_label] if lang2 == lang]
        topfreqb_tuple[lang] = [w for lang2, w in clf_tuple[usage_label] if lang2 == lang]

    for lang in languages:
        print(f"===== {language_names[lang]} =====\n")
        for lang2 in languages:

            swowr123_freq_b = set(topswowr123b[lang]).intersection(set(topfreqb[lang2]));
            swowr123_b = [w for w in topswowr123b[lang] if w in swowr123_freq_b];
            freqr123_b = [w for w in topfreqb[lang2]
                          if w in swowr123_freq_b]

            print("----------------------------")
            print("{} assoc - {} usage correlation".format(lang, lang2))
            print("----------------------------")

            top_number = 100
            overlapped_words = set(topswowr123b_tuple[lang][:top_number]).intersection(set(topfreqb_tuple[lang2][:top_number]))
            correl["nkeywords"][usage_label]["r123"][lang][lang2] = len(overlapped_words)
            correl["nclassified"][usage_label]["r123"][lang][lang2] = len(
                set(topswowr123b[lang]).intersection(set(topfreqb[lang2])))

            if lang != lang2:

                if latex_format:
                    word_lang_lemma = {w: get_lemma_set(lang, w, word_lemma, strong_translations, freq) if lang != "zh" else [tr
                                                                                       for tr in strong_translations[
                                                                                           "en_lemma-{}".format(lang)][
                                                                                           w]]
                                       for w in overlapped_words}
                    word_lang2_lemma = {w: get_lemma_set(lang2, w, word_lemma, strong_translations, freq) if lang2 != "zh" else [tr
                                                                                          for tr in strong_translations[
                                                                                              "en_lemma-{}".format(
                                                                                                  lang2)][w]]
                                        for w in overlapped_words}
                    if lang != "en" and lang2 != "en":
                        print(r"overlapped words ({}): \makecell[tc]{{{}}}".format(len(overlapped_words),
                                                                                   "\\\\".join(
                                                                                       r"\textbf{{{}}}\\ {} ({})\\ {} ({})".format(
                                                                                           w,
                                                                                           "/".join([
                                                                                                        r"\form{{{}}}".format(
                                                                                                            tr) for tr
                                                                                                        in
                                                                                                        word_lang_lemma[
                                                                                                            w]]),
                                                                                           lang,
                                                                                           "/".join([
                                                                                                        r"\form{{{}}}".format(
                                                                                                            tr) for tr
                                                                                                        in
                                                                                                        word_lang2_lemma[
                                                                                                            w]]),
                                                                                           lang2
                                                                                       ) for w in overlapped_words)))

                    elif lang == "en":
                        print(r"overlapped words ({}): \makecell[tc]{{{}}}".format(len(overlapped_words),
                                                                                   "\\\\".join(
                                                                                       r"\textbf{{{}}} (en)\\ {} ({})".format(
                                                                                           w,
                                                                                           "/".join([
                                                                                                        r"\form{{{}}}".format(
                                                                                                            tr) for tr
                                                                                                        in
                                                                                                        word_lang2_lemma[
                                                                                                            w]]),
                                                                                           lang2
                                                                                       ) for w in overlapped_words)))
                    else:
                        print(r"overlapped words ({}): \makecell[tc]{{{}}}".format(len(overlapped_words),
                                                                                   "\\\\".join(
                                                                                       r"\textbf{{{}}} (en)\\ {} ({})".format(
                                                                                           w,
                                                                                           "/".join([
                                                                                                        r"\form{{{}}}".format(
                                                                                                            tr) for tr
                                                                                                        in
                                                                                                        word_lang_lemma[
                                                                                                            w]]),
                                                                                           lang
                                                                                       ) for w in overlapped_words)))
                else:
                    print(f"overlapped words ({len(overlapped_words)}): {overlapped_words}")

            print()

            if lang == lang2:
                low_freq_high_swow = [w for w in sorted(swowr123_b,
                                                        key=lambda w: topswowr123b[lang].index(w) - topfreqb[
                                                            lang].index(w))][:20]
                high_freq_low_swow = [w for w in sorted(swowr123_b,
                                                        key=lambda w: topfreqb[lang].index(w) - topswowr123b[
                                                            lang].index(w))][:20]

                if latex_format:
                    if lang == "zh":
                        low_freq_high_swow = [
                            r"\begin{{otherlanguage*}}{{chinese}}{}\end{{otherlanguage*}} ({})".format(
                                "/".join(strong_translations["en_lemma-zh"][w]), w) for w in low_freq_high_swow]
                        high_freq_low_swow = [
                            r"\begin{{otherlanguage*}}{{chinese}}{}\end{{otherlanguage*}} ({})".format(
                                "/".join(strong_translations["en_lemma-zh"][w]), w) for w in high_freq_low_swow]
                    elif lang != "en":
                        word_lang_lemma = {w: get_lemma_set(lang, w, word_lemma, strong_translations, freq) for w in low_freq_high_swow + high_freq_low_swow}
                        low_freq_high_swow = [
                            "{} ({})".format("/".join([r"\form{{{}}}".format(lem) for lem in word_lang_lemma[w]]), w)
                            for w in low_freq_high_swow]
                        high_freq_low_swow = [
                            "{} ({})".format("/".join([r"\form{{{}}}".format(lem) for lem in word_lang_lemma[w]]), w)
                            for w in high_freq_low_swow]

                    print()
                    print("top 20 high in swow r123, low in freq: {}\n\n".format(", ".join(low_freq_high_swow)))
                    print("top 20 high in freq, low in swow r123: {}\n\n".format(", ".join(high_freq_low_swow)))

                    if lang == "zh":
                        print("{} words in the top {} of both freq and r123: {}".format(len(overlapped_words),
                                                                                        top_number,
                                                                                        ", ".join(["{} ({})".format(
                                                                                            "/".join(
                                                                                                strong_translations[
                                                                                                    "en_lemma-zh"][w]),
                                                                                            w)
                                                                                            for w in
                                                                                            overlapped_words])))
                    elif lang == "en":
                        print(
                            "{} words in the top {} of both freq and r123: {}".format(len(overlapped_words), top_number,
                                                                                      ", ".join(overlapped_words)))
                    else:
                        word_lang_lemma = {w: get_lemma_set(lang, w, word_lemma, strong_translations, freq) for w in overlapped_words}
                        print(
                            "{} words in the top {} of both freq and r123: {}".format(len(overlapped_words), top_number,
                                                                                      ", ".join(["{} ({})".format(
                                                                                          "/".join([
                                                                                                       r"\form{{{}}}".format(
                                                                                                           lem)
                                                                                                       for lem in
                                                                                                       word_lang_lemma[
                                                                                                           w]]), w)
                                                                                                 for w in
                                                                                                 overlapped_words])))
                else:
                    print("top 20 high in swow r123, low in freq: {}\n\n".format(", ".join(low_freq_high_swow)))
                    print("top 20 high in freq, low in swow r123: {}\n\n".format(", ".join(high_freq_low_swow)))
                    if available_keywords is not None:
                        overlapped_words = ["\033[1m{}\033[0m".format(w) if w in available_keywords[lang] else w for w in overlapped_words]
                    print(
                        f"{len(overlapped_words)} words in the top {top_number} of both freq and r123: {overlapped_words}")

        print("\n\n")
    return correl


