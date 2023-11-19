from util import load_strong_translations, expand_chinese_characters
from word_mapping import calculate_frequency, get_wordlist
from statistics import final_score, get_alphas, get_n, sort_tuple_score
import argparse
import pandas as pd
import csv

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--inputf", default=None, type=str, required=True,
                        help="input tsv file with rows of language, untranslated word, frequency")
    parser.add_argument("--strong_translation_dir", default="lemma-strong-translations", type=str, required=False,
                        help="strong translation directory")
    parser.add_argument("--alpha", default=1, type=float, required=False,
                        help='alpha_m^i, smoothing parameter for prior calculation')
    parser.add_argument("--do_not_translate", action="store_true", help="skip translation and use word forms directly")
    parser.add_argument("--outf", default=None, type=str, required=True,
                        help="output file")
    args = parser.parse_args()
    languages = ["zh", "ko", "id", "ms", "en", "nl", "de", "da", "no", "sv", "fi", "lt", "pl", "ru", "uk", "mk", "el",
                 "ro", "it", "fr", "ca", "es", "pt"]

    smoothing_param = args.alpha
    df = pd.read_csv(args.inputf, header=None, na_values=None, na_filter=False, sep='\t', quoting=csv.QUOTE_NONE)
    df = df.astype({0: 'string', 1: 'string', 2: 'float'})
    lines = df.to_dict('split')['data']

    freq = {}
    for line in lines:
        lang, word, fq = tuple(line)
        freq[lang] = freq.get(lang, {})
        freq[lang][word] = fq


    if not args.do_not_translate:
        not_supp_languages = set(freq.keys()) - set(languages)
        if len(not_supp_languages) > 0:
            print(f"{not_supp_languages} not supported and have been excluded from our analysis.\nSupported languages include {languages}.")

        languages = set(languages).intersection(set(freq.keys()))
        strong_translations = load_strong_translations(languages, folder_path=args.strong_translation_dir, source="en_lemma")

        if 'zh' in freq:
            print('Converting traditional Chinese characters to simplified characters ... ')
            freq["zh"] = expand_chinese_characters(freq["zh"], convert="key", count_values=True)

        tr_freq = calculate_frequency(freq, strong_translations, languages, source="en_lemma")

    else:
        languages = set(freq.keys())
        tr_freq = freq

    wordlist = get_wordlist(tr_freq, languages)


    n = get_n(freq)
    alphas = get_alphas(tr_freq, wordlist, smoothing_param)
    bayesian_score = final_score(tr_freq, alphas, wordlist, n)
    bayesian_tuple = sort_tuple_score(bayesian_score)
    bayesian_tuple_rank = {tup: i + 1 for i, tup in enumerate(bayesian_tuple)}

    rankf = open(args.outf, "w")

    if not args.do_not_translate:
        rankf.write("en_lemma\tlanguage\tforms\tfrequency\tsaliency rank\tsaliency score\n")
        for lw, rank in bayesian_tuple_rank.items():
            lang, lemma = lw[0], lw[1]
            forms = set(strong_translations[f'en_lemma-{lang}'][lemma])
            forms = '/'.join([form for form in forms if form in freq[lang]])
            rankf.write("{}\t{}\t{}\t{}\t{}\t{}\n".format(lemma, lang, forms, tr_freq[lang].get(lemma, 'na'), rank, bayesian_score[lang].get(lemma, "na")))
        rankf.close()
        print(f"English lemma, language, forms, frequency, saliency rank, saliency score have been saved to {args.outf}.")
    else:
        rankf.write("form\tlanguage\tfrequency\tsaliency rank\tsaliency score\n")
        for lw, rank in bayesian_tuple_rank.items():
            lang, lemma = lw[0], lw[1]
            rankf.write("{}\t{}\t{}\t{}\t{}\n".format(lemma, lang, tr_freq[lang].get(lemma, 'na'), rank, bayesian_score[lang].get(lemma, "na")))
        rankf.close()
        print(f"Word, language, frequency. saliency rank, saliency score have been saved to {args.outf}.")
    return


if __name__ == "__main__":
    main()