import os
import pprint
import argparse
import numpy as np
from nltk.tokenize import TweetTokenizer
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.feature_extraction import DictVectorizer
from simple_baseline import scoresForStrongBaseline
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import KFold, StratifiedKFold
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from scipy.stats import spearmanr
from scipy.stats import pearsonr
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
import string
from emoint.featurizers.emoint_featurizer import EmoIntFeaturizer
from emoint.ensembles.blending import blend
from sklearn.feature_extraction.text import CountVectorizer




pp = pprint.PrettyPrinter()
parser = argparse.ArgumentParser()
tokenizer = TweetTokenizer()
featurizer = EmoIntFeaturizer()

parser.add_argument('--train_file', type=str, required=True)
parser.add_argument('--bias_data', type=str, required=True)
parser.add_argument('--outputfile', type=str, required=True)

# ADDED FOR DEV
parser.add_argument('--dev_file', type=str, required=True)
parser.add_argument('--outputfiledev', type=str, required=True)

simple_baseline_preds = []

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

    # untokenized version of the tweet, fed into EmoInt
    string_format = ' '.join(tweet).replace(' , ',',').replace(' .','.').replace(' !','!').replace(' ?','?').replace(' : ',': ')

    features = [
        ('tweet', getCleanTweet(tweet))
        # TODO: add more features here.
    ]

    # append elements given by the emoint to features
    emoint_count = 0
    for elmt in featurizer.featurize(string_format, tokenizer):
        features = features + [('emoint' + str(emoint_count), elmt)]
        emoint_count += 1

    return dict(features)

def getTrainfeats(tweet, index):
    """ This takes the word in question and
    the offset with respect to the instance
    word """

      
    # untokenized version of the tweet, fed into EmoInt
    string_format = ' '.join(tweet).replace(' , ',',').replace(' .','.').replace(' !','!').replace(' ?','?').replace(' : ',': ')

    features = [
        #('tweet', getCleanTweet(tweet)),
        ('simple_baseline', getSimpleBaselineScore(index))
        # TODO: add more features here.
    ]

    # append elements given by the emoint to features
    emoint_count = 0
    for elmt in featurizer.featurize(string_format, tokenizer):
        features = features + [('emoint' + str(emoint_count), elmt)]
        emoint_count += 1

    return dict(features)

'''
Run file with python strong_baseline_dev.py --train_file datasets/EI-reg-En-anger-train.txt --dev_file datasets/2018-EI-reg-En-anger-dev.txt --bias_data female.txt --outputfile results/train_preds.txt --outputfiledev results/dev_preds.txt
'''

def main(args):


    all_tweets,train_labels = tokenize_tweets(args.train_file)
    bias_tweets = tokenize_biasdata(args.bias_data)

    # ADDED FOR DEV
    all_tweets_dev, dev_labels = tokenize_tweets(args.dev_file)

    train_feats = []
    test_feats = []

    # ADDED FOR DEV
    dev_feats = [] # will contain the features we use for predicting

    simple_baseline_preds.extend(scoresForStrongBaseline(all_tweets))


    count = 0
    for index, tweet in enumerate(all_tweets):
        feats = getTrainfeats(tweet, index)
        train_feats.append(feats)

    count2 = 0
    for tweet in bias_tweets:
        feats = getTestfeats(tweet)
        test_feats.append(feats)

    # ADDED FOR DEV
    count3 = 0
    for index, tweet in enumerate(all_tweets_dev):
        feats = getTrainfeats(tweet, index)
        dev_feats.append(feats)


    #print(train_feats)
    clf = RandomForestClassifier(n_jobs=2, random_state=0)

    vectorizer = DictVectorizer()

    X_train = vectorizer.fit_transform(train_feats)
    #print(X_train)
    X_test = vectorizer.transform(test_feats)
    #print(X_test)

    # ADDED FOR DEV
    X_dev = vectorizer.transform(dev_feats)


    clf.fit(X_train, train_labels)

    #y_pred = clf.predict(X_train)
    y_pred = clf.predict(X_test)

    with open(args.outputfile, "w+") as outfile:
        for pred in y_pred:
            outfile.write(pred)
            outfile.write("\n")



    # ADDED FOR DEV
    y_pred_dev = clf.predict(X_dev)
    with open(args.outputfiledev, "w+") as outfiledev:
        for pred in y_pred_dev:
            outfiledev.write(pred)
            outfiledev.write("\n")


if __name__ == '__main__':
    args = parser.parse_args()
    pp.pprint(args)
    main(args)
