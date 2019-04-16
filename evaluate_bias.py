import pprint
import argparse
from scipy import stats

pp = pprint.PrettyPrinter()
parser = argparse.ArgumentParser()

parser.add_argument('--file1', type=str, required=True)
parser.add_argument('--file2', type=str, required=True)

def convertFileToList(file):

    predictions = []

    with open(file, 'r') as f:
        text = f.read().strip().split('\n')
        for line in text:
        	predictions.append(line)

    return predictions

def main(args):
    predictions1 = convertFileToList(args.file1)
    predictions2 = convertFileToList(args.file2)

    statistic, pval = stats.ttest_rel(predictions1, predictions2)

    print("T-Statistic: " + str(statistic))
    print("P Value: " + str(pval))

if __name__ == '__main__':
	args = parser.parse_args()
    pp.pprint(args)
    main(args)