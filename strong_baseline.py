import os
import pprint
import argparse
import numpy as np
from nltk.tokenize import TweetTokenizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from simple_baseline import scoresForStrongBaseline
import string

pp = pprint.PrettyPrinter()
parser = argparse.ArgumentParser()

parser.add_argument('--train_file', type=str, required=True)
parser.add_argument('--bias_data', type=str, required=True)
parser.add_argument('--outputfile', type=str, required=True)

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
    features = [
        ('tweet', getCleanTweet(tweet))
        # TODO: add more features here.
    ]
    return dict(features)

def getTrainfeats(tweet, index):
    """ This takes the word in question and
    the offset with respect to the instance
    word """
    features = [
        #('tweet', getCleanTweet(tweet)),
        ('simple_baseline', getSimpleBaselineScore(index))
        # TODO: add more features here.
    ]
    return dict(features)

'''
Run file with python strong_baseline.py --train_file datasets/EI-reg-En-anger-train.txt --bias_data female.txt --outputfile results/train_preds.txt
'''

def main(args):

    all_tweets,train_labels = tokenize_tweets(args.train_file)
    bias_tweets = tokenize_biasdata(args.bias_data)

    train_feats = []
    test_feats = []

    simple_baseline_preds.extend(scoresForStrongBaseline(all_tweets))

    for index, tweet in enumerate(all_tweets):
        feats = getTrainfeats(tweet, index)
        train_feats.append(feats)

    for tweet in bias_tweets:
        feats = getTestfeats(tweet)
        test_feats.append(feats)

    #print(train_feats)

    clf = RandomForestClassifier(n_jobs=2, random_state=0)

    vectorizer = DictVectorizer()
    X_train = vectorizer.fit_transform(train_feats)
    #print(X_train)
    X_test = vectorizer.transform(test_feats)
    #print(X_test)

    clf.fit(X_train, train_labels)

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
