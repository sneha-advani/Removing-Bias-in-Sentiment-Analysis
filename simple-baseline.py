import os
import pprint
import argparse
import numpy as np
from nltk.tokenize import TweetTokenizer

pp = pprint.PrettyPrinter()
parser = argparse.ArgumentParser()

parser.add_argument('--tweet_file', type=str, required=True)
parser.add_argument('--lexicon', type=str, required=True)
parser.add_argument('--outputfile', type=str, required=True)


def lexicon2dict(lexicon):
    #convert affect intensity lexicon to dictionary
    lines_read = 0
    affect_dict = {}
    with open(lexicon, 'r') as f:
        next(f) #skip the header
        for line in f:
            line = line.strip()
            if not line:
                continue
            lines_read += 1
            term, score, affect = line.split("\t")
            affect_dict[term] = np.float(score)
    return affect_dict


def tokenize_tweets(tweet_file):
    #tokenize tweets woot
    lines_read = 0
    tknzr = TweetTokenizer()
    all_tweets = []
    tweet_ids = []
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
            tweet_ids.append(ID)
    return all_tweets, tweet_ids


def get_scores(all_tweets, affect_dict):
    #for each tweet in list, get score
    preds = []
    for tweet in all_tweets:
        tweet_score = 0
        words_found = 0
        for word in tweet:
            try:
                #find average of words found in lexicon
                tweet_score += affect_dict[word]
                words_found += 1
            except KeyError:
                pass
        #create running list of prediction scores
        #print(tweet, words_found, tweet_score)
        if words_found > 0:
            preds.append(tweet_score/words_found)
        else:
            preds.append(-1) #if no words were found in the lexicon, append -1
    return preds

def writePreds2File(preds, tweet_ids, outputfile):
    with open(outputfile, 'w') as f:
        f.write('ID\tscores\n')
        for i in range(len(preds)):
            f.write(tweet_ids[i] + '\t' + str(np.round(preds[i],3))+'\n')
    f.close()
    pass
'''
Run file with python base_model.py --tweet_file datasets/EI-reg-En-anger-train.txt --lexicon datasets/NRC-AffectIntensity-Lexicon.txt  --outputfile simple_baseline_train.txt
'''
def main(args):
    print(args.tweet_file)

    affect_dict = lexicon2dict(args.lexicon)
    all_tweets, tweet_ids = tokenize_tweets(args.tweet_file)
    preds = get_scores(all_tweets, affect_dict)
    writePreds2File(preds, tweet_ids, args.outputfile)

if __name__ == '__main__':
    args = parser.parse_args()
    pp.pprint(args)
    main(args)
