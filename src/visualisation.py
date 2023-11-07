import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import numpy as np
import copy


def tabulate_keyword_ranks(wordlist, final_score, saliency_sc, sorted_saliency_sc, saliency_sc_rank,
                           strong_translations,
                           available_keywords, language_names, source_language, target_languages, languages,
                           highlight_words=None):
    if highlight_words is None:
        prev_highlighted_words = ["please", "patient", "fasting", "lord", "heart", "amen", "god", "mother", "wrong" \
            , "hope", "homeland", "greetings", "eid", "happy", "pride", "lazy", "tea", "loyal", \
                                  "christmas", "carnival", "bath", "parade"]
        possible_highlighted_words = set(wordlist).intersection(set(prev_highlighted_words))
        possible_highlighted_ids = [sorted_saliency_sc.index(hl) for hl in list(possible_highlighted_words)]
        possible_keywords = [w for words in available_keywords.values() for w in words]
        possible_keywords_ids = [sorted_saliency_sc.index(wo) for wo in possible_keywords]
        highlighted_word_ids = list(range(0, 10)) + possible_highlighted_ids + possible_keywords_ids + \
                               [len(sorted_saliency_sc) - 1]
    else:
        highlighted_word_ids = [sorted_saliency_sc.index(wo) for wo in highlight_words]

    ids = sorted(set(highlighted_word_ids))

    # red words appear in Table 1 from the paper
    # candidate keywords are underlined
    # words with highest score (rank) are bold

    BOLD = '\033[1m'
    END = '\033[0m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    UNDERLINE = '\033[4m'

    print("Ranked keywords by class score with facebook mapping")
    language_str = ""
    for l in target_languages + [source_language]:
        language_str += (language_names[l] + ", ")
    print("rank, class score, {}".format(language_str))

    for id in ids:
        w = sorted_saliency_sc[id]
        r = saliency_sc_rank[w]
        s = round(saliency_sc[w], 2)
        max_lang = max(final_score.keys(), key=lambda l: final_score.get(l, {}).get(w, -999))
        all_w_str = []
        for i, l in enumerate(languages):
            if w in final_score[l]:

                if l == source_language:
                    w_str = copy.deepcopy(w)
                else:
                    w_str = "/".join(copy.deepcopy(strong_translations[source_language + "-" + l][w]))

                if w in available_keywords[l]:
                    w_str = UNDERLINE + w_str + END

                if l == max_lang:
                    w_str = BOLD + w_str + END

                all_w_str.append(w_str)

            else:
                all_w_str.append("N/A")

        if highlight_words is None and w in possible_highlighted_words:
            all_w_str = [RED + wo + END for wo in all_w_str]

        all_w_str_as_one = ""
        for t in all_w_str:
            all_w_str_as_one += (t + ", ")

        print("{} {} {}".format(r, s, all_w_str_as_one))

    print()



def compare_recall_ranks_all_methods_inset(method_cat_bin_ratio_dict, method_style_dict, method_label_dict,
                                           method_colour_dict, top_n=None, inset_range=[0, 5000, 0.01, 0.20], fill_between=None):
    fig, ax = plt.subplots(figsize=[10, 6])

    method0 = list(method_style_dict.keys())[0]
    top_n = len(method_cat_bin_ratio_dict[method0]) if top_n is None else top_n

    bins = (np.array(list(range(0, len(method_cat_bin_ratio_dict[method0]), 1))) + 1)

    for method, sty in method_style_dict.items():
        plt.plot(bins, method_cat_bin_ratio_dict[method], sty, label=method_label_dict[method],
                 color=method_colour_dict[method])
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.ylabel('Recall', fontsize=18)
    plt.xlabel('Top Ranks', fontsize=18)
    # plt.title('Recall @ Ranks for Correctly Identified Keywords',fontweight="bold", size=18)
    plt.legend(loc="best", prop={'size': 15})

    if fill_between is not None:
        for method in method_style_dict:
            plt.fill_between(bins, fill_between[method][0], fill_between[method][1], color=method_colour_dict[method],
                             alpha=0.4)

    if inset_range is not None:

        sub_axes = plt.axes([0.7, 0.15, .15, .5])
        for method, sty in method_style_dict.items():
            sub_axes.plot(bins[:inset_range[1]], method_cat_bin_ratio_dict[method][:inset_range[1]], sty,
                          label=method_label_dict[method], color=method_colour_dict[method])
            if fill_between is not None:
                sub_axes.fill_between(bins[:inset_range[1]], fill_between[method][0][:inset_range[1]],
                                      fill_between[method][1][:inset_range[1]], color=method_colour_dict[method],
                                      alpha=0.4)
        sub_axes.set_title(f"Top {inset_range[1]} words")
        sub_axes.get_xaxis().set_visible(False)