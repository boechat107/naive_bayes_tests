FROM ipython/scipystack

RUN pip2 install -U nltk

RUN ipython2 -c "import nltk; nltk.download('movie_reviews')"
