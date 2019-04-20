## Description of Evaluation Metrics:
We have two scripts that we will use to evaluate the performance of our model on both the SemEval Tweet Corpus and on the Equity Evaluation Corpus. 

### Evaluation on "SemEval Affect in Tweets Corpus":
Use the evaluate_tweets.py script to measure performance of our sentiment analysis model on the original task test set. NOTE: evaluate tweets has a dependency on utils.py, so make sure that file is present in the directory as well. 
This metric is the Pearson Correlation Coefficient - a higher score is better as it indicates that the two 
sets provided are closer in value. 
```
python evaluate_tweets.py <file-predictions> <file-gold> 
```

### Evaluation on "Equity Evaluation Corpus":
Use the evaluate_bias.py script to measure the amount of bias the sentiment analysis model has when predicting on two groups of people. The script requires two input files, each containing a prediction of the intensity of an emotion on every line. Both files must contain the same amount of predictions, and the two files should be the predictions of the sentiment analysis models on the same sentences from the Equity Evaluation Corpus, simply with the first file having the templates filled in with names corresponding to one group (for example, male names) and the other file containing templates filled in with names corresponding to another group (for example, female names). After running the script, the t-statistic and p value found from doing a paired t-test on the two sets of predictions are computed and printed. 

```
python evaluate_bias.py --file1 <file-1> --file2 <file-2>
```