#!/usr/bin/env python2
# -*- coding: utf-8 -*-

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# user_evaluation.py
# Author: felipebravom
# Descrition: Command-line version of the evaluation script for SemEval-2018 Task 1: Affect in Tweets
# usage: python evaluate.py <task_type> <file-predictions> <file-gold> 
# task_type: 1 for regression, 2 for ordinal classification, and 3 for multi-label emotion classification
# requires: numpy, scipy, sklearn


import sys
import os.path
import scipy.stats
import numpy as np

def main(argv):
    """main method """   
    
    if len(argv)!=2:
        raise ValueError('Invalid number of parameters.')


    pred=argv[0]
    gold=argv[1] 

    result=evaluate_ei(pred,gold)
    print("Pearson correlation between "+os.path.basename(pred)+" and "+os.path.basename(gold)+":\t"+str(result[0]))
    print("Pearson correlation for gold scores in range 0.5-1 between "+os.path.basename(pred)+" and "+os.path.basename(gold)+":\t"+str(result[1])) 


def evaluate_ei(pred,gold):  
    """Calculates performance metrics for regression.
    
    :param pred: the file path of the predictions
    :param gold: the filte path withe gold data
    :return: a list with performace metrics.
    
    """
    
    f=open(pred, "rb")
    pred_lines=f.readlines()
    f.close()
    
    f=open(gold, "rb")
    gold_lines=f.readlines()
    f.close()

    if(len(pred_lines)==len(gold_lines)):       
        # align tweets ids with gold scores and predictions
        data_dic={}


        header=True        
        for line in gold_lines:
            line=line.decode('utf-8')
            
            if header:
                header=False
                continue

            parts=line.split('\t')
            if len(parts)==4:
                # tweet ids containing the word mystery are discarded
                if(not 'mystery' in parts[0]):
                    data_dic[parts[0]]=[float(line.split('\t')[3])]
            else:
                sys.exit('Format problem in '+os.path.basename(gold)+'. Please report this problem to the task organizers.')
                
 

        header=True        
        for line in pred_lines:
            line=line.decode('utf-8')
            if header:
                header=False
                continue

            parts=line.split('\t')
            if len(parts)==2:
                # tweet ids containing the word mystery are discarded
                if(not 'mystery' in parts[0]):
                    if parts[0] in data_dic:
                        try:
                            data_dic[parts[0]].append(float(line.split('\t')[1]))
                        except ValueError:
                            # Invalid predictions are replaced by a default value
                            data_dic[parts[0]].append(0.5)
                    else:
                        sys.exit('Invalid tweet id ('+parts[0]+') in '+os.path.basename(pred)+'.')
            else:
                sys.exit('Format problem in '+os.path.basename(pred)+'.') 
            
        
        # lists storing gold and prediction scores
        gold_scores=[]  
        pred_scores=[]
         
        
        # lists storing gold and prediction scores where gold score >= 0.5
        gold_scores_range_05_1=[]
        pred_scores_range_05_1=[]
            
        for id in data_dic:
            if(len(data_dic[id])==2):
                gold_scores.append(data_dic[id][0])
                pred_scores.append(data_dic[id][1])
                
                if(data_dic[id][0]>=0.5):
                    gold_scores_range_05_1.append(data_dic[id][0])
                    pred_scores_range_05_1.append(data_dic[id][1])
                
            else:
                sys.exit('Repeated id ('+id+') in '+os.path.basename(pred)+' .')

                
                    
      
        # return zero correlation if predictions are constant
        if np.std(pred_scores)==0 or np.std(gold_scores)==0:
            return (0,0)
        

        pears_corr=scipy.stats.pearsonr(pred_scores,gold_scores)[0]                                                 
                                    
        
        pears_corr_range_05_1=scipy.stats.pearsonr(pred_scores_range_05_1,gold_scores_range_05_1)[0]                         
        
      
        return (pears_corr,pears_corr_range_05_1)
       
                                    
                          
        
    else:
        sys.exit('Predictions ('+os.path.basename(pred)+') and gold data ('+os.path.basename(gold)+') have different number of lines.')

if __name__ == "__main__":
    main(sys.argv[1:])
