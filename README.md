## Computing Word Importance (Saliency) Scores
1. `pip install numpy pandas`
2. Unzip `lemma-strong-translations.zip`
3. Run `python src/meaning_importance.py --inputf <input file> --strong_translation_dir lemma-strong-translations-0.2 --outf test_saliency.tsv`. This will translate word forms and produce output file `test_saliency.tsv`.  Note that `<input file>` is a tab separated file (without header) that includes language (corpus), word form and frequency. 
4. You can add `--do_not_translate` flag when comparing usage of same word forms across languages (corpora). This will skip all preprocesses involving translations.

## Evaluated keywords
Revealing keywords are documented in `goddard-wierzbicka.csv`
Unrevealing keywords are documented in `external-keywords.csv`

## Results and analysis

1. Results and analysis reported in Section 5 are shown in notebook `word-usage-analysis-V4c.ipynb`.
Results and analysis reported in Section 6 are shown in notebook `word-assoc-analysis-V4c.ipynb`.
2. There are three types of files in `measurements`:
   1. `*_word_importance_score*.txt` records scores of meaning classes associated with respective language (Equation 1, 
   described in Section 5.1).
   2. `*_word_saliency*.txt` records saliency scores of meaning classes, corresponding to Section 5.3. 
   3. `*_word_classification*.txt` records language classification (rank=1) of meaning classes based on 
   word importance scores above.



## Strong translations
`lemma-strong-translations.zip` include multilingual words (including inflected forms) to English lemma 
mappings, readily preprocessed with the steps below. The rules and detailsare described in 
Supplementary 1.1 and 1.2.

To get inexact English-French translations by strong translation rules (Supplementary 1.1)
1. Download [WorldLex Data](http://worldlex.lexique.org/files/Fre.Freq.2.rar), unzip the files
2. Download [fastText aligned word vectors](https://fasttext.cc/docs/en/aligned-vectors.html)
3. `pip3 install OpenCC nltk`
4. `python3 src/strong_translate.py --lang fr --fasttext_emb_dir <aligned vectors directory> --freq_dir <worldlex data directory> --output_dir <output directory>`

To gather morphological variants based on strong translations in French (Supplementary 1.2)
1. Create directories: `mkdir -p lemmatized mean_lemma_emb lemma-strong-translations`
2. Lemmatize all words: `python3 src/word_lemmatize.py --spacy_model <spacy_model> --lang fr --output_file lemmatized/fr.txt --bsz 2000 --n_process 1`
3. Mean embeddings for each lemma cluster: `python3 src/mean_fasttext_emb.py --aligned_vectors_dir <vectors directory> --lemma_dir lemmatized --output_dir mean_lemma_emb --lang $language`
4. Translate all morphological variants and store the translations in `lemma-strong-translations`: `python3 src/lemma_strong_translate.py --lemma_vectors_dir mean_lemma_emb --lemma_dir lemmatized 
 --strong_translations_dir <strong translation directory (from above)> --source_lang en
 --output_dir lemma-strong-translations --lang fr
 --aligned_vectors_dir fasttext_emb --source_spacy_model <spacy_model> --sim_thresh 0.2`

## Small World of Words

https://smallworldofwords.org/en/project/home





