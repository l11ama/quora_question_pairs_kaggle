#Competition
**Quora question pairs**

https://www.kaggle.com/c/quora-question-pairs

# Main solution

**Key moments:**
* It's unacceptable to make classification on all pairs with such heavy classifier as BERT in real life. I would use BERT to get good encodings for questions.
* Train dataset has [selection bias](https://arxiv.org/abs/1905.06221), seems private set also has this issue. I prefer not to use these leakage features to improve the score. But such train
dataset leads to overfitting problem, which's still unresolved by me.
 ![alt-text-leakage](imgs/leakage_vis-1.png) ![alt-text-q1q2](imgs/q1q2_vis-1.png)
* To make BERT \<CLS\> encodings more suitable for final task I try to finetune them with metric learning on triplets. Also this procedure helps with selection bias problem in case of good triplet generator.
* Token embeddings are also important to train a good model. To save information from not \<CLS\> tokens I use classifier head with extra input: sum of all token embeddings attended to other question.
* I didn't use any ensambles, more interesting for me was to do experiments with embedding learning with single heavy model.

**Requirements**
- check `requirements.txt` and install missing packages
- download a pre-trained BERT model and place it in this folder, specify the chosen model in `config.yml` (`path_to_pretrained_model`). For example, for a medium uncased model that will be: 
- `wget -q https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip'`
- `unzip uncased_L-12_H-768_A-12.zip`
- place your CSV files in the input folder (`path_to_data` in `models/bert_finetuning/config.yml`)
- specify batch size in `models/bert_finetuning/config.yml` and `models/bert_finetuning/config_triplets.yml`:
- install apex or change `apex_mixed_precision` to `False`
- debug with option `toy=True`, get real submission with `toy=False` 

**Run whole training process and get submission file**
```
python submission.py
```
## Phase 1: Metric Learning

### Create triplet dataset from question pairs
**[notebook](notebooks/bert_finetuning/0_0_data_preprocessing_triplets.ipynb)**

* split train pair dataset with stratification on `train.csv` data (validation part = 0.1)
* build positive and negative connection graphs on train set 
* collect buckets with duplicated questions
* detect all negative connections from each bucket to other buckets 
* generate triplets: for each pair of questions in bucket - 3 negative examples
 

### Train metric learning   

**[notebook](notebooks/bert_finetuning/0_1_bert_metric_learning_triplet_loss.ipynb)**

* encode anchor, positive and negative questions from triplet with BERT separately
* train with [Triplet Loss](https://arxiv.org/abs/1412.6622) for 3 encoded \<CLS\> tokens respectively

![alt text](imgs/metric_learning.svg)

## Phase 2: Pair classifier

**[notebook](notebooks/bert_finetuning/0_2_bert_finetuning_pair_clf_inner_prod_attention_with_metric_learning.ipynb)**

* split train pair dataset in same way as for metric learning to reduce data leak
* load metric learned encoder BERT encoder
* encode left and right question with BERT separately
* pool all encoded tokens from one question attended to encoded \<CLS\> token of another question
* concat attended embedding with \<CLS>\ embedding for each question
* get elementwise product of question embedding
* make binary classification with 2-layer classifier's head
* freezing BERT layers for first epoch, different learning rate for head and BERT layers  

![alt text](imgs/pair_classifier.svg)

## Metrics

**Kaggle link**: https://www.kaggle.com/mfside

* implemented in `submission.py`: 2 epochs metric learning + 1 epoch pair clf with freezed BERT + 3 epochs pair clf with unfreezed BERT.
[Model on Google Drive](https://drive.google.com/open?id=1Yf0kEFhlt3JnlKgZgXlUW29IZLKGOimK)

    `Private = 0.38092  Public = 0.37776`
    
* finetune one more epoch with decreased lr. [Model on Google Drive](https://drive.google.com/open?id=1_f-7Rg73adwtOtXTjrrhhs_fm3UFgLQG)

    `Private = 0.37830  Public = 0.37406`
    
* finetune another one epoch with decreased lr and weighted loss (try to increase precision, balance of duplicate question in test is lower) [Model on Google Drive](https://drive.google.com/open?id=17RcXYzcO9kfqy4kdL8yzPFA6YXxWz8XD)

    `Private = 0.36893 Public = 0.36465`
    
## Conclusions

**Main unresolved problem: _overfitting_.**

 BERT model learns well, but on large epoches overfit effect could be recognized both on validation and test set.
 
_Ways to resolve:_
 - detailed data analysis. Influence of selection bias, explore ways to select representative validation datasets.  
 - make metric learning more efficient to generalize model better (hard to train, but could be optimized with 
 [N-Pair Loss](https://papers.nips.cc/paper/6200-improved-deep-metric-learning-with-multi-class-n-pair-loss-objective.pdf))
 - change sampling from train dataset for pair classification (class imbalance, test class imbalance). Simple method with weighted loss gives better results (on last experiment).
 - hyper parameters tuning, reduce model capacity (DisillBERT, less amount of linear layers, reduced hidden states). Not have enough time for experiments, hard to train without good GPUs.
  
**Make experiments more clear**

- Detect influence of each phase and architecture part and take the best variant

# Other methods

### BERT finetuning on pair of question joined by separator

**[notebook](notebooks/bert_finetuning/1_0_bert_finetuning_separator.ipynb)**

**[Model on Google disk](https://drive.google.com/open?id=18sR-2ZxeMQlv516dhrByPru-ctA_Mpzz)**

`Private = 0.33768 Public = 0.33507`

### ULMFiT

**[notebook](notebooks/ulmfit/0_quora_question_pairs_ulmfit_classifier_dssm_att_fc1.ipynb)**

**[vocab on Google disk](https://drive.google.com/open?id=1ulopED1wsUFogG6B0IsKf4Ia1DXW13eS)**

**[Model on Google disk](https://drive.google.com/open?id=1_Ww01R8IFonIyDpU4rSOYnELLyuZ3T_x)**

`Private = 0.36800 Public = 0.36714`
