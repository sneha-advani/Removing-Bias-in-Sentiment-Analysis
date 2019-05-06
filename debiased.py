import os
import pprint
import argparse
import numpy as np
from nltk.tokenize import TweetTokenizer
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from scipy.stats import spearmanr
from scipy.stats import pearsonr
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
import string
from sklearn.feature_extraction.text import CountVectorizer
import gensim

pp = pprint.PrettyPrinter()
parser = argparse.ArgumentParser()

parser.add_argument('--train_file', type=str, required=True)
parser.add_argument('--bias_data', type=str, required=True)
parser.add_argument('--outputfile', type=str, required=True)

model = gensim.models.KeyedVectors.load_word2vec_format('datasets/GoogleNews-vectors-negative300-hard-debiased.txt', binary=False)  

def tokenize_tweets(tweet_file):
    #tokenize tweets woot
    lines_read = 0
    tknzr = TweetTokenizer(preserve_case=False)
    all_tweets = []
    labels = []

    #read file, tokenize all tweets, append to list
    with open(tweet_file, 'r', encoding='utf8') as f:
        next(f) #skip the header
        for line in f:
            line = line.strip()
            if not line:
                continue
            lines_read += 1
            ID, tweet, affect, score = line.split("\t")
            words = tknzr.tokenize(tweet)
            all_tweets.append(words)
            labels.append(score)
    return all_tweets, labels


def tokenize_biasdata(tweet_file):
    #tokenize EEC bias_data tweets
    #tokenized from file output of 'extractData.py'

    lines_read = 0
    tknzr = TweetTokenizer(preserve_case=False)
    all_tweets = []

    #read file, tokenize all tweets, append to list
    with open(tweet_file, 'r', encoding='utf8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            lines_read += 1
            ID, tweet = line.split("\t")
            words = tknzr.tokenize(tweet)
            all_tweets.append(words)
    return all_tweets

def getCleanTweet(tweet):
    tweet_str = " ".join(tweet)
    for symbol in string.punctuation:
        if symbol in tweet_str:
            tweet_str = tweet_str.replace(symbol, "")
    return tweet_str

def getSimpleBaselineScore(index):
    return simple_baseline_preds[index]

def getTestfeats(tweet):
    """ This takes the word in question and
    the offset with respect to the instance
    word """
    features = [
    ]
    words = [np.zeros(300)]
    for word in tweet:
#       print(word)
      try:
        words.append(model[word])
#         print(word + " -- found")
      except Exception as e:
#         print(word + " -- notfound")
        print(e)
        continue
        
    average = np.mean(words, axis=0)
    for i in range(300):
        features = features + [('debias' + str(i), average[i])]
        
    return dict(features)

def getTrainfeats(tweet, index):
    """ This takes the word in question and
    the offset with respect to the instance
    word """
    features = [
    ]
    
    words = [np.zeros(300)]
    for word in tweet:
      try:
        words.append(model[word])
#         print(word + " -- found")
      except Exception as e:
#         print(word + " -- notfound")
        print(e)
        continue
        
    average = np.mean(words, axis=0)

    for i in range(300):
        features = features + [('debias' + str(i), average[i])]
        
    return dict(features)

def main(args):

    all_tweets,train_labels = tokenize_tweets(args.train_file)
#     bias_tweets = tokenize_biasdata(bias_data)
    dev_tweets,dev_labels = tokenize_tweets(args.bias_data)
  
    train_feats = []
    test_feats = []

    for index, tweet in enumerate(all_tweets):
        print(tweet)
        feats = getTrainfeats(tweet, index)
        train_feats.append(feats)
    
    for index, tweet in enumerate(dev_tweets):
        print(tweet)
        feats = getTestfeats(tweet)
        test_feats.append(feats)

#     for tweet in bias_tweets:
#         feats = getTestfeats(tweet)
#         test_feats.append(feats)


    clf = RandomForestClassifier(n_jobs=2, random_state=0)

    vectorizer = DictVectorizer()
    X_train = vectorizer.fit_transform(train_feats)
    X_test = vectorizer.transform(test_feats)

    clf.fit(X_train, train_labels)
    
    print("FINISHED TRAINING")

    #y_pred = clf.predict(X_train)
    y_pred = clf.predict(X_test)

    with open(args.outputfile, "w+") as outfile:
        for pred in y_pred:
            outfile.write(pred)
            outfile.write("\n")

if __name__ == '__main__':
    args = parser.parse_args()
    pp.pprint(args)
    main(args)
