# Modified by: Coco Sittardt
# Date: 01.04.2026
# Changes: new preprocessing version for sentence-based tokens
# Date: 07.04.2026
# Changes: simple method for tokenizing sentences
import re

from nltk import sent_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from stop_words import get_stop_words
import nltk
from collections import Counter
from typing import Tuple, List

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

en_stop = get_stop_words('en')
sw = stopwords.words("english")


stop_words = sw + en_stop
stop_words.append('let')
stop_words.append('gon')
stop_words.append('dhe')
stop_words.extend(['like', 'got',
                   'get', 'one', 'well',
                   'back', 'bit', 'drive',
                   'look', 'see', 'good',
                   'quite', 'think', 'little',
                   'right', 'know', 'thing', 'want'])
stop_words.extend(['put', 'yeah', 'lot''dot', 'le', "'ve", 'really', 'like', 'got', 'get', 'one', 'well',
                   'back', 'bit', 'drive', 'look', 'see', 'good', 'quite', 'think', 'little', 'right', 'know',
                   'thing', 'want', 'dhe', 'gon', 'let', 'get'])
stop_words.extend(["\'re", "n\'t", "n\'t", "'ve", "really", "car", "cars"])


def preprocessing(segments: list, segment_labels: list, preprocessing_type: str, graph_level: str="words") \
        -> Tuple[list, list, list, list]:
    """
    preprocessing is used to preprocess the data set
    
    :param segments: raw list of segments
    :param segment_labels: list of segment labels
    :param preprocessing_type: defines the preprocessing approach (["JN", "FP"])
    :param graph_level: activates sentence level processing if set to sentences

    :return:
        - preprocessed segments
        - labels of preprocessed segments
        - sorted list of vocabulary words
        - list of tokenized 'raw' segments

    """
    do_stemming = False

    if preprocessing_type == "MUSE":
        # preprocessing for MUSE
        do_just_nouns = True
        do_lemmatizing = False
        do_stop_word_removal = False
        remove_low_freq = False
        count_threshold = 1
    else:
        assert preprocessing_type == "CRR"
        # preprocessing for CRR
        do_just_nouns = False
        do_lemmatizing = False
        do_stop_word_removal = True
        remove_low_freq = True
        count_threshold = 30

    # set everything to False if graph level is sentences to not remove any tokens
    if graph_level == "sentences":
        do_just_nouns, do_lemmatizing, do_stop_word_removal, remove_low_freq= False, False, False, False
        count_threshold = 1

    vocabulary = []
    new_docs = []
    new_labels = []
    tokenized_docs = []
    for i, doc in enumerate(segments):
        if graph_level == "words":
            doc = doc.lower()
            tokens = word_tokenize(doc)

            # remove all tokens that are < 3
            tokens = [w for w in tokens if len(w) > 2]

            # remove all tokens that are just digits
            tokens = [w for w in tokens if w.isalpha()]

            tokenized_doc = [w for w in tokens]

            # remove stop words before stemming/lemmatizing
            if do_stop_word_removal:
                tokens = [w for w in tokens if w not in stop_words]

            # remove all words that are not nouns
            if do_just_nouns:
                tokens = [w for (w, pos) in nltk.pos_tag(tokens) if pos in ['NN', 'NNP', 'NNS', 'NNPS']]

            # stemming
            if do_stemming:
                tokens = [PorterStemmer().stem(w) for w in tokens]

            # lemmatizing
            if do_lemmatizing:
                tokens = [WordNetLemmatizer().lemmatize(w) for w in tokens]

            if len(tokens) == 0:
                continue

        # [Coco] for sentences, it is not needed to do any of the lemmatizing or stemming
        else:
            assert graph_level == "sentences"
            tokens = sent_tokenize(doc)
            tokenized_doc = [w for w in tokens]
            if len(tokens) == 0:
                continue


        new_docs.append(tokens)
        new_labels.append(segment_labels[i])
        vocabulary.extend(tokens)
        tokenized_docs.append(tokenized_doc)




    if remove_low_freq:
        # remove low-frequency terms

        temp_all_data = []
        for d in new_docs:
            temp_all_data.extend(d)
        counter = Counter(temp_all_data)

        docs_threshold = []
        labels_threshold = []
        vocab_threshold = []
        for i_d, d in enumerate(new_docs):

            d_threshold = [w for w in d if counter[w] > count_threshold]
            if len(d_threshold) > 0:

                labels_threshold.append(new_labels[i_d])
                docs_threshold.append(d_threshold)
                vocab_threshold.extend(d_threshold)

        new_docs = docs_threshold
        vocabulary = vocab_threshold
        new_labels = labels_threshold

    assert len(new_docs) == len(new_labels)
    return new_docs, new_labels, sorted(list(set(vocabulary))), tokenized_docs


def get_tokenized_sentences(tokenized_docs: list)\
    -> List[List[str]]:
    """
    [Coco]
    Method to get tokenized versions of sentences. Uses basic regex to split and strip it
    :param tokenized_docs: list of sentences
    :return: list of lists of tokens per sentence
    """
    tokenized_word_docs = []
    for chunk in tokenized_docs:
        for sentence in chunk:
            doc = sentence.lower().split()
            # regexing the punctuation
            doc = [re.sub(r'[^\w\s]', '', w) for w in doc]
            # and removing the empty tokens ones from the regex
            doc = [w for w in doc if w]
            tokenized_word_docs.append(doc)
    return tokenized_word_docs

