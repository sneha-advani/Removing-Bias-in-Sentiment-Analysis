import os
import pprint
import argparse

pp = pprint.PrettyPrinter()
parser = argparse.ArgumentParser()

parser.add_argument('--inputeecfile', type=str, required=True)
parser.add_argument('--gender', type=str, required=False)
parser.add_argument('--race', type=str, required=False)
parser.add_argument('--outputfile', type=str, required=True)

'''
Extract hypo hyper names  from the inputwikifile and given the hearstpatterns class
'''

def extractData(inputeecfile, gender, race):
    '''Each line in inputeec file contains 8 columns, including
        col1: sentence
        col5: gender (male/female)
        col6: race (African-American / European / None)
        col7: emotion
    '''

    # Should contain list of (hyponym, hypernym) tuples
    extractions = []

    with open(inputeecfile, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            line_split = line.split(",")
            thisGender = line_split[4].strip()
            thisRace = line_split[5].strip()
            if (race == None or thisRace.lower() == race.lower()) and (gender == None or gender.lower() == thisGender.lower()):
                extractions.append(line)

    return extractions


'''
Writes the hypo hyper pairs to output file
'''
def writeDataToFile(dataList, outputfile):
    directory = os.path.dirname(outputfile)
    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(outputfile, 'w') as f:
        for line in dataList:
            f.write(line + '\n')

'''

Sample use: python extractData.py --inputeecfile filename.csv --gender male --race african-american --outputfile ./out.csv

'''
def main(args):

    dataList = extractData(args.inputeecfile, args.gender, args.race)

    writeDataToFile(dataList, args.outputfile)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
