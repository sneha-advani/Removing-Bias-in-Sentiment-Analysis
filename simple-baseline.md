Description of the Simple Baseline:
Our baseline takes an input tweet, tokenizes the tweet, and compares 
the words to our Affect Intensity Lexicon. The affect intensity Lexicon 
includes a list of words associated with anger and include a certain score.
For each tweet, we look up how many of the words are in the lexicon, 
get the scores for those words, and take the average. If no words
from the tweet are present in the lexicon, we assign a -1 for the score. 

The output of our baseline model on our training set was 
.362 and on our dev set was .314. 