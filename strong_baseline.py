import os
import pprint
import argparse
import numpy as np
from sklearn.linear_model import Ridge


# from sklearn.feature_extraction import DictVectorizer
# from simple_baseline import scoresForStrongBaseline
# from scipy.stats import spearmanr
# from scipy.stats import pearsonr
# from sklearn.metrics import accuracy_score
# import string
# from sklearn.feature_extraction.text import CountVectorizer

pp = pprint.PrettyPrinter()
parser = argparse.ArgumentParser()

#parser.add_argument('--feature_files', type=str, required=True)
parser.add_argument('--train_feats', type=str, nargs='+')
parser.add_argument('--test_feats', type=str, nargs='+')
parser.add_argument('--outputfile', type=str, required=True)

def get_feats(feat_files):
    all_feats = []
    for f in feat_files:
        all_feats.append([np.float(line) for line in open(f, 'r').readlines()])
    return all_feats

def train_model()

'''
Run file with python strong_baseline.py --feature_files feature_outputs/emoint_predictions_train_male.txt feature_out
puts/emoint_predictions_train_female.txt --outputfile female_predictions.txt

'''

def main(args):
    X_train = get_feats(args.train_feats)
    X_test = get_feats(args.test_feats)




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
