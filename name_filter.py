
import nltk
import argparse
import pprint
from nltk.tokenize import TweetTokenizer

parser = argparse.ArgumentParser()
pp = pprint.PrettyPrinter()

parser.add_argument('--input_file', type=str, required=True)
parser.add_argument('--names', type=str, required=True)
parser.add_argument('--outputfile', type=str, required=True)

def tokenize_tweets(tweet_file):
    tknzr = TweetTokenizer(preserve_case=False)
    all_tweets = []

    #read file, tokenize all tweets, append to list
    with open(tweet_file, 'r', encoding='utf8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ID, tweet = line.split("\t")
            words = tknzr.tokenize(tweet)
            all_tweets.append(words)
    return all_tweets

def get_names(names_file):
    all_names = []

    with open(names_file, 'r', encoding='utf8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            all_names.append(line)
    return all_names

def replace_words(tweets, names):
    lowercase = [x.lower() for x in names]
    gendered_pronouns = ["he", "she", "him", "her", "his", "hers", "himself", "herself"]
    neutral_pronouns = ["they", "they", "them", "their", "their", "theirs", "themself", "themself"]
    for tweet in tweets:
        for i, elt in enumerate(tweet):
            if elt in lowercase:
                tweet[i] = "[Name]"
            if elt in gendered_pronouns:
                tweet[i] = neutral_pronouns[gendered_pronouns.index(elt)]

def write_to_file(file, tweets):
    with open(args.outputfile, "w+") as outfile:
        for tweet in tweets:
            string_format = ' '.join(tweet).replace(' , ',',').replace(' .','.').replace(' !','!').replace(' ?','?').replace(' : ',': ')
            outfile.write(string_format)
            outfile.write("\n")

def main(args):
    tweets = tokenize_tweets(args.input_file)
    replace_words(tweets, get_names(args.names))
    write_to_file(args.outputfile, tweets)


#python3 name_filter.py --input_file input.txt --names names.txt --outputfile out.txt

if __name__ == '__main__':
    args = parser.parse_args()
    pp.pprint(args)
    main(args)