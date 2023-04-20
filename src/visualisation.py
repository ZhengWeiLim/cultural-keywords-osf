import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import numpy as np
import copy

# def plot_concepts_by_categories(category_conc_rank, category_mean_rank, category_max_conc_rank,
#                                 category_min_conc_rank, category_optimal_rank):
#     fig, ax = plt.subplots(figsize=(17, 10))
#
#     plt.title("Expected Keyword Ranks by Categories", fontsize="xx-large")
#     categories = list(category_conc_rank.keys())
#     no_categories = len(categories)
#     xs = range(0, no_categories)
#     ys = [[r for conc, rs in conc_rank.items() for r in rs] for cat, conc_rank in category_conc_rank.items()]
#
#     for x, cat in enumerate(categories):
#         #     ax.plot([x], [category_max_conc_rank[cat][1]], "ko")
#         max_conc = category_max_conc_rank[cat][0]
#         max_conc = max_conc.replace(" ", "\n") if len(max_conc) > 12 else max_conc
#         ax.annotate(max_conc, (x, category_max_conc_rank[cat][1]), textcoords='offset points')
#         #     ax.plot([x], [category_min_conc_rank[cat][1]], "ko")
#         min_conc = category_min_conc_rank[cat][0]
#         min_conc = min_conc.replace(" ", "\n") if len(min_conc) > 12 else min_conc
#         ax.annotate(min_conc, (x, category_min_conc_rank[cat][1]), textcoords='offset points')
#         ax.plot([x], [category_mean_rank[cat]], "r_", markersize=10)
#         ax.plot([x], [category_optimal_rank[cat]], "b_", markersize=10)
#
#         for rs in category_conc_rank[cat].values():
#             for r in rs:
#                 ax.plot([x], [r], "ko", alpha=0.3)
#
#     ax.plot(xs, [4] * len(xs), "k:", label="rank by chance", markersize=10)
#     ax.plot([], [], "r_", label="mean rank", markersize=10)
#     ax.plot([], [], "b_", label="optimal mean rank", markersize=10)
#
#     ax.legend()
#     plt.xticks(xs, categories, fontsize="large")
#     plt.xlabel("category", fontsize="large")
#     plt.ylabel("rank", fontsize="large")


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


# histogram-like
# method options: bayesian, proportion, descending, ascending and bayesian_cluster
# def plot_by_rank_bins_categories(method, bin_sz, nwords, method_cat_bin_ratio_dict):
#     bins = range(0, nwords, bin_sz)
#     str_bins = [str(i) if len(str(i)) >= 5 else str(i)+"  " for i in bins]
#     str_bins[0] = "<"+str_bins[1]
#     str_bins[-1] = r"$\geq$"+str_bins[-1]
#     str_bins[1:-1] = [str(bins[i+1])+"-\n"+str(b) for i, b in enumerate(bins[2:])]
#     x_bin = np.array(range(len(bins))) * 4
#     width = 0.8
#
#     fig, ax = plt.subplots(figsize=(15,6))
#
#     p1 = ax.bar(x_bin, method_cat_bin_ratio_dict[method]["en-ext"]["range"],
#                 width, label='external (en)', color="lightskyblue")
#     p2 = ax.bar(x_bin+width, method_cat_bin_ratio_dict[method]["non-en-ext"]["range"],
#                 width, label='external (non-en)', color="royalblue")
#     p3 = ax.bar(x_bin+2*width, method_cat_bin_ratio_dict[method]["en-int"]["range"],
#                 label="internal (en)", width=width, color="bisque")
#     p4 = ax.bar(x_bin+3*width, method_cat_bin_ratio_dict[method]["non-en-int"]["range"],
#                 label="internal (non-en)", width=width, color="orange")
#
#
#     ax.set_ylabel('Ratio of keywords', fontsize=12)
#     ax.set_xlabel('Rank', fontsize=12)
#     ax.set_title('Ratio of Correctly Identified Keywords by Saliency Ranks with {} Model'.format(method.capitalize()),
#                  fontweight="bold", size=14)
#     plt.xticks(x_bin+1.5*width, str_bins, fontsize=10, horizontalalignment="center")
#     ax.legend(loc="best", prop={'size': 11})


# Mean and spread of bayesian method (bar graph)
# def plot_mean_spread_bar(method, en_stats_language, non_en_stats):
#     '''
#     method: bayesian, proportion, descending, ascending and bayesian_cluster
#     en_stats_language: {method: {mean/std/recall: {external/internal/all: {language: value}}}
#     non_en_stats: {method: {mean/std/recall: {external/internal/all: value}}}
#     '''
#     x_bin = np.array(range(3))
#     str_bins = ["external", "internal", "all"]
#     width = 0.3
#     fig, ax = plt.subplots(figsize=(10,5))
#
#     en_mean_ranks = [en_stats_language[method]["mean"]["external"]["en"],
#                      en_stats_language[method]["mean"]["internal"]["en"],
#                      en_stats_language[method]["mean"]["all"]["en"]]
#     en_error = [en_stats_language[method]["std"]["external"]["en"],
#                 en_stats_language[method]["std"]["internal"]["en"],
#                 en_stats_language[method]["std"]["all"]["en"]]
#     non_en_mean_ranks = [non_en_stats[method]["mean"]["external"],
#                          non_en_stats[method]["mean"]["internal"],
#                          non_en_stats[method]["mean"]["all"]]
#     non_en_error = [non_en_stats[method]["std"]["external"],
#                     non_en_stats[method]["std"]["internal"],
#                     non_en_stats[method]["std"]["all"]]


    # p1 = ax.bar(x_bin, en_mean_ranks, yerr=en_error, width=width, label='English',
    #             color="lightskyblue",capsize=5)
    # p2 = ax.bar(x_bin+width, non_en_mean_ranks, yerr=non_en_error, width=width, label='Non-English',
    #             color="royalblue",capsize=5)
    #
    # ax.set_ylabel('Mean rank', fontsize=12)
    # ax.set_xlabel('Keyword', fontsize=12)
    # ax.set_title('Mean Rank of Gold Standard Keywords \nwith {} Model'.format(method.capitalize()),
    #              fontweight="bold", size=14)
    # plt.xticks(x_bin+width/2, str_bins, fontsize=10, horizontalalignment="center")
    # ax.legend(loc="best", prop={'size': 11})


# compare 2 methods and their recall@ranks by categories (line graph)
# def compare_recall_ranks_categories(method1, method2, bin_sz, method_cat_bin_ratio_dict, top_n=None):
#
#     top_n = len(method_cat_bin_ratio_dict[method1]["en-ext"]["sum"]) if top_n is None else top_n
#
#     bins = (np.array(list(range(0, len(method_cat_bin_ratio_dict[method1]["en-ext"]["sum"]), bin_sz))) + 1)[:top_n]
#
#     plt.figure(figsize=(12,6))
#
#     plt.plot(bins, method_cat_bin_ratio_dict[method1]["en-ext"]["sum"][:top_n],
#              "b-")
#     plt.plot(bins, method_cat_bin_ratio_dict[method1]["en-int"]["sum"][:top_n],
#              "b-.")
#     plt.plot(bins, method_cat_bin_ratio_dict[method1]["non-en-ext"]["sum"][:top_n],
#              "b:")
#     plt.plot(bins, method_cat_bin_ratio_dict[method1]["non-en-int"]["sum"][:top_n],
#              "b--")
#
#     plt.plot(bins, method_cat_bin_ratio_dict[method2]["en-ext"]["sum"][:top_n],
#              "r-")
#     plt.plot(bins, method_cat_bin_ratio_dict[method2]["en-int"]["sum"][:top_n],
#              "r-.")
#     plt.plot(bins, method_cat_bin_ratio_dict[method2]["non-en-ext"]["sum"][:top_n],
#              "r:")
#     plt.plot(bins, method_cat_bin_ratio_dict[method2]["non-en-int"]["sum"][:top_n],
#              "r--")
#     plt.plot([],[],"b-",label=method1)
#     plt.plot([],[],"r-",label=method2)
#     plt.plot([],[],"k--",label="internal (non-en)")
#     plt.plot([],[],"k-.",label="internal (en)")
#     plt.plot([],[],"k:",label="external (non-en)")
#     plt.plot([],[],"k-",label="external (en)")
#
#     plt.ylabel('Recall', fontsize=12)
#     plt.xlabel('Top Ranks', fontsize=12)
#     plt.title('Recall @ Ranks for Correctly Identified Keywords given Keyword Categories',fontweight="bold", size=14)
#     plt.legend(loc="best", prop={'size': 11})
#
#
# def compare_recall_ranks_all_methods(method_cat_bin_ratio_dict, top_n=None,
#                                      methods=["bayesian", "proportion", "ascending", "descending", "bayesian_cluster"]):
#     top_n = len(method_cat_bin_ratio_dict["bayesian"]["all"]["sum"]) if top_n is None else top_n
#
#     bins = (np.array(list(range(0, len(method_cat_bin_ratio_dict["bayesian"]["all"]["sum"]), 1))) + 1)[:top_n]
#     plt.figure(figsize=(12,6))
#
#     for method in methods:
#         if method == "bayesian":
#             plt.plot(bins, method_cat_bin_ratio_dict["bayesian"]["all"]["sum"][:top_n],
#                  "r-",label='Bayesian (word)', color="tab:red")
#         elif method == "proportion":
#             plt.plot(bins, method_cat_bin_ratio_dict["proportion"]["all"]["sum"][:top_n],
#                      "k--",label='proportion-based', color="tab:blue")
#         elif method == "ascending":
#             plt.plot(bins, method_cat_bin_ratio_dict["ascending"]["all"]["sum"][:top_n],
#                     "k-.",label="ascending frequency", color="tab:blue")
#         elif method == "descending":
#             plt.plot(bins, method_cat_bin_ratio_dict["descending"]["all"]["sum"][:top_n],
#                     "k:", label="descending frequency",  color="tab:blue")
#         elif method == "bayesian_cluster":
#             plt.plot(bins, method_cat_bin_ratio_dict["bayesian_cluster"]["all"]["sum"][:top_n],
#                      "r--",label='Bayesian (cluster)', color="tab:red")
#
#     plt.ylabel('Recall', fontsize=12)
#     plt.xlabel('Top Ranks', fontsize=12)
#     plt.title('Recall @ Ranks for Correctly Identified Keywords',fontweight="bold", size=14)
#     plt.legend(loc="best", prop={'size': 11})


# def tabulate_keyword_ranks_by_cluster(wordlist, word_cluster, clusters,
#                                       final_score, clt_saliency_sc, clt_sorted_saliency_sc, word_sc_rank,
#                                       strong_translations, available_keywords, non_en_keywords,
#                                       language_names, source_language,
#                                       target_languages, languages, highlight_words=None):
#     if highlight_words is None:
#         prev_highlighted_words = ["please", "patient", "fasting", "lord", "heart", "amen", "god", "mother", "wrong",
#                                   "hope", "homeland", "greetings", "eid", "happy", "pride", "lazy", "tea", "loyal",
#                                   "christmas", "carnival", "bath", "parade"]
#         possible_highlighted_words = set(wordlist).intersection(set(prev_highlighted_words))
#         possible_highlighted_ids = [clt_sorted_saliency_sc.index(word_cluster[hl])
#                                     for hl in list(possible_highlighted_words)]
#         possible_keywords = [w for words in available_keywords.values() for w in words]
#         possible_keywords_ids = [clt_sorted_saliency_sc.index(word_cluster[wo]) for wo in possible_keywords]
#         highlighted_word_ids = list(range(0, 10)) + possible_highlighted_ids + possible_keywords_ids + \
#                                [len(clt_sorted_saliency_sc) - 1]
#     else:
#         highlighted_word_ids = [clt_sorted_saliency_sc.index(word_cluster[wo]) for wo in highlight_words]
#
#     ids = sorted(set(highlighted_word_ids))
#
#     # red words appear in Table 1 from the paper
#     # candidate keywords are underlined
#     # words with highest score (rank) are bold
#
#     BOLD = '\033[1m'
#     END = '\033[0m'
#     RED = '\033[91m'
#     BLUE = '\033[94m'
#     UNDERLINE = '\033[4m'
#
#     print("Ranked keywords by class score with facebook mapping")
#     language_str = ""
#     for l in target_languages + [source_language]:
#         language_str += (language_names[l] + ", ")
#     print("rank, class score, {}".format(language_str))
#
#     for id in ids:
#         w = clt_sorted_saliency_sc[id]
#         r = word_sc_rank[list(list(clusters[w].values())[0])[0]]
#         s = round(clt_saliency_sc[w], 2)
#         max_sc = -999
#         max_idx = None
#         all_w_str = []
#         for i, l in enumerate(languages):
#             if w in final_score[l]:
#                 w_sc = final_score[l][w]
#
#                 if w_sc > max_sc:
#                     max_sc = w_sc
#                     max_idx = i
#                 if l == source_language:
#                     target_words = copy.deepcopy(list(clusters[w][l]))
#                     target_words = [UNDERLINE + tw + END if tw in (available_keywords[l]) else tw
#                                     for tw in target_words]
#                 else:
#                     target_words = copy.deepcopy(list(clusters[w][l]))
#                     translations = [tr for kw in available_keywords[l] for tr in
#                                     strong_translations["en-{}".format(l)][kw]
#                                     if kw in strong_translations["en-{}".format(l)]]
#                     target_words = [UNDERLINE + tw + END
#                                     if tw in (non_en_keywords[l]) or tw in translations else tw
#                                     for tw in target_words]
#
#                 w_str = "/".join(target_words)
#
#                 all_w_str.append(w_str)
#             else:
#                 all_w_str.append("N/A")
#
#         all_w_str[max_idx] = BOLD + all_w_str[max_idx] + END
#
#         if highlight_words is None and len(set(clusters[w]["en"]).intersection(set(possible_highlighted_words))) > 0:
#             all_w_str = [RED + wo + END for wo in all_w_str]
#
#         all_w_str_as_one = ""
#         for t in all_w_str:
#             all_w_str_as_one += (t + ", ")
#
#         print("{} {} {}".format(r, s, all_w_str_as_one))


# def compare_recall_ranks_all_methods_inset(method_cat_bin_ratio_dict, method_style_dict, method_label_dict,
#                                            method_colour_dict, top_n=None, inset_range=[0, 5000, 0.01, 0.20],
#                                            percent=10, fill_between=None):
#     fig, ax = plt.subplots(figsize=[10, 6])
#
#     method0 = list(method_style_dict.keys())[0]
#     top_n = len(method_cat_bin_ratio_dict[method0]["all"]["sum"]) if top_n is None else top_n
#
#     bins = (np.array(list(range(0, len(method_cat_bin_ratio_dict[method0]["all"]["sum"]), 1))) + 1)
#
#     for method, sty in method_style_dict.items():
#         plt.plot(bins, method_cat_bin_ratio_dict[method]["all"]["sum"], sty, label=method_label_dict[method],
#                  color=method_colour_dict[method])
#     plt.xticks(fontsize=13)
#     plt.yticks(fontsize=13)
#     plt.ylabel('Recall', fontsize=18)
#     plt.xlabel('Top Ranks', fontsize=18)
#     # plt.title('Recall @ Ranks for Correctly Identified Keywords',fontweight="bold", size=18)
#     plt.legend(loc="best", prop={'size': 15})
#
#     if fill_between is not None:
#         for method in method_style_dict:
#             plt.fill_between(bins, fill_between[method][0], fill_between[method][1], color=method_colour_dict[method],
#                              alpha=0.4)
#
#     if inset_range is not None:
#         axins = zoomed_inset_axes(ax, 1.7, loc=4, bbox_to_anchor=(0.4, 0.05, .5, .3),
#                                   bbox_transform=ax.transAxes)
#
#         for method, sty in method_style_dict.items():
#             axins.plot(bins[:top_n], method_cat_bin_ratio_dict[method]["all"]["sum"][:top_n], sty,
#                        label=method_label_dict[method], color=method_colour_dict[method])
#
#         axins.axis(inset_range)
#         axins.set_yticks([])
#         axins.set_xticks([])
#         axins.set_title('Top {}% ranks'.format(percent), fontsize=18)
#         plt.yticks(visible=False)
#         plt.xticks(visible=False)
#
#         mark_inset(ax, axins, loc1=4, loc2=2, fc="none", ec="0.0")


def compare_recall_ranks_all_methods_inset(method_cat_bin_ratio_dict, method_style_dict, method_label_dict,
                                           method_colour_dict, top_n=None, inset_range=[0, 5000, 0.01, 0.20],
                                           percent=10, fill_between=None, inset_out=False):
    fig, ax = plt.subplots(figsize=[10, 6])

    method0 = list(method_style_dict.keys())[0]
    top_n = len(method_cat_bin_ratio_dict[method0]["all"]["sum"]) if top_n is None else top_n

    bins = (np.array(list(range(0, len(method_cat_bin_ratio_dict[method0]["all"]["sum"]), 1))) + 1)

    for method, sty in method_style_dict.items():
        plt.plot(bins, method_cat_bin_ratio_dict[method]["all"]["sum"], sty, label=method_label_dict[method],
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
        bbox_to_anchor = (0.4, 0.05, 0.85, .1) if inset_out else (0.4, 0.05, .5, .3)
        axins = zoomed_inset_axes(ax, 1.7, loc=4, bbox_to_anchor=bbox_to_anchor,
                                  bbox_transform=ax.transAxes)

        for method, sty in method_style_dict.items():
            axins.plot(bins[:top_n], method_cat_bin_ratio_dict[method]["all"]["sum"][:top_n], sty,
                       label=method_label_dict[method], color=method_colour_dict[method])

        if fill_between is not None:
            for method in method_style_dict:
                axins.fill_between(bins[:top_n], fill_between[method][0][:top_n], fill_between[method][1][:top_n],
                                   color=method_colour_dict[method],
                                   alpha=0.4)

        axins.axis(inset_range)
        axins.set_yticks([])
        axins.set_xticks([])
        axins.set_title('Top {}% ranks'.format(percent), fontsize=18)
        plt.yticks(visible=False)
        plt.xticks(visible=False)

        mark_inset(ax, axins, loc1=4, loc2=2, fc="none", ec="0.0")
    return axins


