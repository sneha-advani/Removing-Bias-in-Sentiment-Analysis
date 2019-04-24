# Removing-Bias-in-Sentiment-Analysis

Instructions for running strong baseline:

```
python strong_baseline.py --train_file datasets/EI-reg-En-anger-train.txt --bias_data female.txt --outputfile results/train_preds.txt
```

This will output the predicted results (taking in the sentences from female.txt) to a file called train_preds.txt.
