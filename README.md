# AMR-PLM

# Experimental Environments
+ Python 3.8
+ torch 1.8.1
+ apex 0.1

```
  cd spring
  pip install -e .
  pip install -r requirments.txt
```

# Training

## AMR2Text

```
  cd amr-seq2seq
  bash finetune_bart_amr2text_large.sh facebook/bart-large
```

## AMRParsing

```
  cd amr-seq2seq
  bash finetune_bart_amrparsing_large.sh facebook/bart-large
```
