import os
import pprint
import argparse
import numpy as np
from nltk.tokenize import TweetTokenizer
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.feature_extraction import DictVectorizer
from simple_baseline import scoresForStrongBaseline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from scipy.stats import spearmanr
from scipy.stats import pearsonr
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
import string
from emoint.featurizers.emoint_featurizer import EmoIntFeaturizer
from emoint.ensembles.blending import blend
from sklearn.feature_extraction.text import CountVectorizer
import skipthoughts
from encoder import Model




pp = pprint.PrettyPrinter()
parser = argparse.ArgumentParser()
tokenizer = TweetTokenizer()
featurizer = EmoIntFeaturizer()
model = Model()

parser.add_argument('--train_file', type=str, required=True)
parser.add_argument('--bias_data', type=str, required=True)
parser.add_argument('--outputfile', type=str, required=True)

simple_baseline_preds = []

skipthoughts_model = skipthoughts.load_model()
skipthoughts_encoder = skipthoughts.Encoder(skipthoughts_model)

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
    text_features = model.transform([string_format])
    
    features = [
        ('tweet', getCleanTweet(tweet))
        # TODO: add more features here.
    ]

    # append elements given by the emoint to features
    emoint_count = 0
    for elmt in featurizer.featurize(string_format, tokenizer):
        features = features + [('emoint' + str(emoint_count), elmt)]
        emoint_count += 1
        
    sentiment_count = 0
    for ft in text_features:
        features = features + [('sentiment' + str(sentiment_count), np.sqrt(ft.dot(ft)))]
        sentiment_count += 1
        
    skipthought_count = 0
    average = np.zeros(4800)
    for i in skipthoughts_encoder.encode(tweet):
        average += i
        skipthought_count += 1
    average = average / skipthought_count

    for i in range(4800):
        features = features + [('skipthought' + str(i), average[i])]

    return dict(features)

def getTrainfeats(tweet, index):
    """ This takes the word in question and
    the offset with respect to the instance
    word """

      
    # untokenized version of the tweet, fed into EmoInt
    string_format = ' '.join(tweet).replace(' , ',',').replace(' .','.').replace(' !','!').replace(' ?','?').replace(' : ',': ')
    text_features = model.transform([string_format])

    features = [
        # ('tweet', getCleanTweet(tweet)),
        ('simple_baseline', getSimpleBaselineScore(index))
        # TODO: add more features here.
    ]

    # append elements given by the emoint to features
    emoint_count = 0
    for elmt in featurizer.featurize(string_format, tokenizer):
        features = features + [('emoint' + str(emoint_count), elmt)]
        emoint_count += 1

    sentiment_count = 0
    for ft in text_features:
        features = features + [('sentiment' + str(sentiment_count), np.sqrt(ft.dot(ft)))]
        sentiment_count += 1

    skipthought_count = 0
    average = np.zeros(4800)
    for i in skipthoughts_encoder.encode(tweet):
        average += i
        skipthought_count += 1
    average = average / skipthought_count

    for i in range(4800):
        features = features + [('skipthought' + str(i), average[i])]

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


    count = 0
    for index, tweet in enumerate(all_tweets):
        feats = getTrainfeats(tweet, index)
        train_feats.append(feats)

    count2 = 0
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
