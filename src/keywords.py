import csv
import nltk
from nltk.corpus import stopwords

# from expected_keywords.txt
def load_external_keywords():
    expected_keywords = list(csv.reader(open("external-keywords.csv", 'r'), delimiter=','))

    concept_words_language = {} # {concept: {"word": words, "language": langs, "category": category}}

    lang_keyword = {}

    for row in expected_keywords:
        concept = row[0]
        words = row[0].split("/")
        words = [w.lower() for mwe in words for w in mwe.split(" ")]
        category = row[1]
        langs = row[2].split("/")
        concept_words_language[concept] = {"word": words, "language": langs, "category": category}
        for lang in langs:
            if lang in lang_keyword:
                lang_keyword[lang] = set(lang_keyword[lang]).union(set(words))
            else:
                lang_keyword[lang] = set(words)

    return concept_words_language, lang_keyword


# from goddard-wierzbicka.csv
def load_goddard_wierzbicka_keywords(split=True):
    cultural_keywords = list(csv.reader(open("goddard-wierzbicka.csv", 'r'), delimiter=','))

    lang_keyword = {}

    for row in cultural_keywords[1:]:
        words = row[0].split("/") if split else [row[0]]
        words = [w for mwe in words for w in mwe.split(" ")]
        for word in words:
            if row[1] in lang_keyword:
                lang_keyword[row[1]].add(word)
            else:
                lang_keyword[row[1]] = set([word])

    return lang_keyword # {language: set(words)}


def combine_keyword_sets(lang_keyword_list):
    lang_keyword = {}
    for lang_keyword_dict in lang_keyword_list:
        for lang, items in lang_keyword_dict.items():
            if lang in lang_keyword:
                lang_keyword[lang] = set(lang_keyword[lang]).union(set(items))
            else:
                lang_keyword[lang] = set(items)
    return lang_keyword


def translate_keywords(lang_keyword, bilingual_dict, source="en"):
    # translate to source/reference language
    lang_keyword_source = {}

    for lang, keyword in lang_keyword.items():
        if lang not in lang_keyword_source:
            lang_keyword_source[lang] = set([])

        if lang == source:
            lang_keyword_source[lang] = lang_keyword_source[lang].union(set(keyword))
        else:
            for word in keyword:
                if word in bilingual_dict[lang+"-"+source]:
                    trans = bilingual_dict[lang+"-"+source][word]
                    trans = [trans] if isinstance(trans, str) else trans
                    lang_keyword_source[lang] = lang_keyword_source[lang].union(set(trans))

    return lang_keyword_source


def retrieve_all_available_keywords(wordlist, lang_keyword, word_freq=None, stop_words=stopwords.words('english')):
    available_lang_keyword = {}

    for lang, keywords in lang_keyword.items():
        if word_freq is not None:
            available_lang_keyword[lang] = set(wordlist).intersection(keywords).intersection(set(word_freq[lang].keys())) - set(stop_words)
        else:
            available_lang_keyword[lang] = set(wordlist).intersection(keywords) - set(stop_words)

    return available_lang_keyword


def filter_strong_translations(lang_keyword, strong_translations, source="en"):
    filtered_lang_keyword = {}
    for lang, keywords in lang_keyword.items():
        if lang == source:
            filtered_lang_keyword[lang] = set(keywords)
        else:
            filtered_lang_keyword[lang] = set(keywords).intersection(set(strong_translations["{}-{}".format(source, lang)].keys()))
    return filtered_lang_keyword


# cultural_keywords: cultural keywords from goddard-wierzbicka.txt before translation,
# expected_keywords: wikipedia keywords assumed to be english from external-keywords.csv
# concept_words_language from expected_keywords.txt, separated by concepts
#
# after strong translations:
# all_keywords_en: combination of translated cultural keywords and expected keywords
# available_keywords: above, exclude non-comparable keywords (exist in < 2 languages)
# internal_keywords & external_keywords: break down of the above into internal and external cultural
# aspects of keywords
def load_and_translate_keywords(strong_translations, languages, source, wordlist, word_freq=None):
    cultural_keywords = load_goddard_wierzbicka_keywords()

    # expected keywords from external_keywords.txt
    concept_words_language, expected_keywords, = load_external_keywords()

    for language in list(cultural_keywords.keys()):
        if language not in languages:
            cultural_keywords.pop(language)

    for language in list(expected_keywords.keys()):
        if language not in languages:
            expected_keywords.pop(language)

    # translate cultural keywords to english via strong translations
    translated_keywords = translate_keywords(cultural_keywords, strong_translations, source)

    # combine and retrieve list of all possible keywords
    all_keywords_en = combine_keyword_sets([expected_keywords, translated_keywords])
    all_keywords_en = filter_strong_translations(all_keywords_en, strong_translations, source)
    available_keywords = retrieve_all_available_keywords(wordlist, all_keywords_en, word_freq)
    # available_keywords = filter_strong_translations(available_keywords, strong_translations)

    # external vs internal keywords
    external_keywords = {l: kwords.intersection(available_keywords[l]) for l, kwords in expected_keywords.items()}
    internal_keywords = {l: set() for l in languages}

    for lang in languages:
        if lang in translated_keywords:
            internal_keywords[lang] = internal_keywords[lang].union(
                translated_keywords[lang]).intersection(available_keywords[lang])

    return cultural_keywords, concept_words_language, expected_keywords,\
           all_keywords_en, available_keywords, internal_keywords, external_keywords
