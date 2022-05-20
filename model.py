import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
import gensim
import gensim.corpora as corpora
from gensim.models.phrases import Phraser
from gensim.models.phrases import Phrases
from gensim.models import word2vec
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
import pyLDAvis
import pyLDAvis.gensim_models
import matplotlib.pyplot as plt
import glob
import os
import numpy as np
import io

stopwords = stopwords.words('english')
bad_words = ['Empty', 'DataFrame', 'Columns']

extension = 'txt'
lemmatizer = WordNetLemmatizer()
all_filenames = [i for i in glob.glob(r'C:\Users\chris\OneDrive\Desktop\lda_example\*.{}'.format(extension))]

def clean_sent(df):
    for file in all_filenames:
        df = pd.read_csv(file)
        words = nltk.word_tokenize(str(df))
        bag = []
        for word in words:
            lemmad = lemmatizer.lemmatize(word)
            if len(lemmad) > 1 and lemmad not in stopwords and lemmad not in bad_words:
                bag.append(lemmad)
    return(bag)

new_data = clean_sent(all_filenames)

out_str = io.StringIO(' '.join(new_data))
df = pd.read_csv(out_str)

def final_out(sentence):
    for sentence in df:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))

text_output = list(final_out(df))

id2word = corpora.Dictionary(text_output)
texts = text_output
corpus = [id2word.doc2bow(text) for text in texts]

lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=id2word, 
num_topics=5, random_state=100, update_every=1, 
chunksize=100, passes=5, alpha='auto')

file_topics = lda_model.show_topics(num_topics=10, log=False, formatted=True)

topic_columns = []
allowed_topics = ['revenue', 'expense', 'cash', 'asset', 'cost', 'stock', 'customer']
top_list = ('\n'.join([str(x) for t in file_topics for x in t])).split('"')
for topic in top_list:
    if topic.isalpha():
        if topic not in topic_columns and topic in allowed_topics:
            topic_columns.append(topic)

word_rows = []
for word in new_data:
    if word.isalpha():
        if word not in word_rows:
            word_rows.append(word)
