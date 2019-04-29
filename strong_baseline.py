import os
import pprint
import argparse
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

pp = pprint.PrettyPrinter()
parser = argparse.ArgumentParser()

#parser.add_argument('--feature_files', type=str, required=True)
parser.add_argument('--train_file', type=str)
parser.add_argument('--train_feats', type=str, nargs='+')
parser.add_argument('--test_feats', type=str, nargs='+')
parser.add_argument('--outputfile', type=str, required=True)

def get_feats(feat_files):
    all_feats = []
    for f in feat_files:
        all_feats.append([np.float64(line) for line in open(f, 'r').readlines()])
    return np.transpose(np.array(all_feats))


def get_train_y(train_file):
    y = []
    # read file, tokenize all tweets, append to list
    with open(train_file, 'r', encoding='utf8') as f:
        next(f)  # skip the header
        for line in f:
            line = line.strip()
            if not line:
                continue
            _, _, _, score = line.split("\t")
            y.append(np.float(score))
    return np.array(y)


def run_model(X_train, y_train):
    pass


'''
Run file with
python strong_baseline.py --train_file datasets/EI-reg-En-anger-train.txt --train_feats feature_outputs/emoint_train_predictions.txt feature_outputs/simple_baseline_train.txt --test_feats feature_outputs/emoint_dev_predictions.txt feature_outputs/simple_baseline_dev.txt --outputfile final_predictions/dev_predictions.txt
'''

def main(args):
    X_train = get_feats(args.train_feats)
    X_test = get_feats(args.test_feats)
    y_train = get_train_y(args.train_file)

    print(X_train.shape, y_train.shape, X_test.shape)

    model = Ridge()
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    with open(args.outputfile, "w+") as outfile:
        for pred in y_pred:
            outfile.write(str(np.round(pred,3)))
            outfile.write("\n")
    outfile.close()

    pass

if __name__ == '__main__':
    args = parser.parse_args()
    pp.pprint(args)
    main(args)
