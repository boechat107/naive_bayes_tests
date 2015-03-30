from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_selection import SelectKBest, chi2
import nltk.classify as nltk
from nltk.corpus import movie_reviews
from nltk.probability import FreqDist
import random


## List of documents where each one is a tuple (words, category) and 'words' is 
## a list too.
documents = [(movie_reviews.words(fid), cat)
             for cat in movie_reviews.categories()
             for fid in movie_reviews.fileids(cat)]
random.shuffle(documents)


def most_frequent_words(n):
    """
    Returns the n most frequent words of the corpus.
    """
    freq_words = FreqDist(w.lower() for w in movie_reviews.words())
    most = freq_words.most_common()[:n]
    return map(lambda (w, freq): w, most)


def words_featureset(words_doc, ref_feats):
    dfreq_ws = FreqDist(w.lower() for w in words_doc)
    featureset = {}
    for w in ref_feats:
        if dfreq_ws.has_key(w):
            featureset[w] = dfreq_ws[w]
    return featureset


def docs_featureset(docs, ref_feats):
    return [(words_featureset(d, ref_feats), c) for (d, c) in docs]


def train_classifier(feats):
    #pipe = Pipeline([('tfidf', TfidfTransformer()),
    #                ('chi2', SelectKBest(chi2, k=1000)),
    #                ('nb', MultinomialNB())])
    classifier = nltk.SklearnClassifier(MultinomialNB())
    classifier.train(feats)
    return classifier


if __name__ == "__main__":
    features = most_frequent_words(2000)
    classifier = train_classifier(docs_featureset(documents[100:], features))
    print nltk.accuracy(classifier, docs_featureset(documents[:100], features))
