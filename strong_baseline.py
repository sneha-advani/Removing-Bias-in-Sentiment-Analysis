import os
import pprint
import argparse
import numpy as np
from nltk.tokenize import TweetTokenizer

pp = pprint.PrettyPrinter()
parser = argparse.ArgumentParser()

parser.add_argument('--train_file', type=str, required=True)
parser.add_argument('--bias_data', type=str, required=True)
parser.add_argument('--outputfile', type=str, required=True)


def tokenize_tweets(tweet_file):
    #tokenize tweets woot
    lines_read = 0
    tknzr = TweetTokenizer(preserve_case=False)
    all_tweets = []

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
    return all_tweets


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
'''
Run file with python strong_baseline.py --train_file datasets/EI-reg-En-anger-train.txt --bias_data female.txt --outputfile results/train_preds.txt
'''

def main(args):

    all_tweets = tokenize_tweets(args.train_file)
    bias_tweets = tokenize_biasdata(args.bias_data)
    print(all_tweets)

if __name__ == '__main__':
    args = parser.parse_args()
    pp.pprint(args)
    main(args)
