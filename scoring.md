## Description of Evaluation Metrics:
We have two scripts that we will use to evaluate the performance of our model on both the SemEval Tweet Corpus and on the Equity Evaluation Corpus. 

### Evaluation on "SemEval Affect in Tweets Corpus":
Use the evaluate_tweets.py script to measure performance of our sentiment analysis model on the original task test set. NOTE: evaluate tweets has a dependency on utils.py, so make sure that file is present in the directory as well. 

```python
evaluate_tweets.py <file-predictions> <file-gold> 
```

### Evaluation on "Equity Evaluation Corpus":
