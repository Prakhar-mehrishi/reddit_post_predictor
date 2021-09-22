import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
import os
from text_processing import clean_text_regex, remove_stop_words, stem_clean_text
from text_processing import t, d, a

curr_path = os.path.dirname(os.path.realpath(__file__))

with open(curr_path + "/resources/small.joblib", 'rb') as s:
    small = joblib.load(s)

with open(curr_path + "/resources/medium.joblib", 'rb') as m:
    medium = joblib.load(m)

with open(curr_path + "/resources/large.joblib", 'rb') as l:
    large = joblib.load(l)

knn_final = joblib.load(curr_path + "/resources/knn_final.joblib")
rf_final = joblib.load(curr_path + "/resources/rf_final.joblib")

with open(curr_path + "/resources/vector.joblib", 'rb') as vect:
    vector = joblib.load(vect)

def get_thumbnail(thumbnail:str):

    i = t.index(thumbnail)
    return pd.get_dummies(t).iloc[i].tolist()

def get_domain(domain:str):

    i = d.index(domain)

    return pd.get_dummies(d).iloc[i].tolist()


def label_reddit(subreddit):
    if subreddit in small:
        return 0
    
    return 1 if subreddit in medium else 2

def Binarize_comments(num):
    if num > 2:
        return 1
    return 0


def polarity_score(text):
    
    text = clean_text_regex(text)
    text = remove_stop_words(text)

    token = RegexpTokenizer("\w+|\$[\d\d]+|http\S+")
    sia = SIA()
    lemm = WordNetLemmatizer()
    pos = 0
    neg = 0
    for word in token.tokenize(text):
        word = lemm.lemmatize(word)
        pos += sia.polarity_scores(word)["pos"]
        neg += sia.polarity_scores(word)["neg"]
    
    return (pos-neg)


def create_matrix(text:str):

    text = clean_text_regex(text)
    text = remove_stop_words(text)
    text = stem_clean_text(text)

    return vector.transform([text]).toarray()[0]

def predict_pop(attributes:list):
    # [0 subreddit, 1 gilded, 2 BinarisedNum_Comments, 3 Binarised_over_18, 4 Title, 5 Subtext, 6 weekday, 7 hour, 8 domain, 9 thumbnail]
    subreddit = label_reddit(attributes[0])

    gilded = int(attributes[1])

    binary_comment = Binarize_comments(attributes[2])
    
    binary_is_self = 1

    over_18_tag = attributes[3]

    polarity = polarity_score(attributes[4] + "\n" + attributes[5])

    matrix = create_matrix(attributes[4] + "\n" + attributes[5])

    thumbnail = get_thumbnail(attributes[9])

    domain = get_domain(attributes[8])

    week_day = attributes[6]

    hour_ = attributes[7].hour

    
    input_feats = np.array([subreddit, gilded, binary_comment, binary_is_self, over_18_tag, polarity])

    thumbnail_domain_feats = np.concatenate((np.array(thumbnail), np.array(domain)))

    final_inp = np.concatenate((input_feats, matrix, thumbnail_domain_feats, np.array([week_day, hour_])))

    prediction = knn_final.predict(final_inp.reshape(1,-1))[0]
    # prediction = knn_final.predict(np.array(a).reshape(1,-1))[0]
    # print(final_inp)
    return prediction




