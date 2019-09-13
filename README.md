# Multi-label Classification (Pytorch)
- Repo for MLC (Multi-label Classification) models.


## Support [Datasets](https://drive.google.com/file/d/18-JOCIj9v5bZCrn9CIsk23W4wyhroCp_/view?usp=sharing)
* RCV1-V2
* AAPD

## Support Models
* NMT
* SGM (TODO)
* ...

## Requirements
* Python 3.6 +
* Pytorch 1.0.0 +
* requirements.txt
* Folder organization
```
MLC
│   README.md
│   embeddings.py
│   model.py
│   run.py
│   utils.py
│   vocab.py
│   requirements.txt
│   
└───data
│   └───AAPD
│   │    │   aapd.json
│   │    │   label_test
│   │    │   label_train
│   │    │   label_val
│   │    │   text_test
│   │    │   text_train
│   │    │   text_val
│   │    
│   └───RCV1-V2
│        │   same as AAPD
│       
└───checkpoints
│   └───AAPD
│   │    │   model.bin
│   │    │   model.bin.optim
│   │
│   │    
│   └───RCV1-V2
│        │   same as AAPD
│        
└───outputs
    └───AAPD
    │    │   test_outputs.txt
    │
    │    
    └───RCV1-V2
         │   same as AAPD
    
```

## Building Vocablary
```
python vocab.py -train-src=<file> -train-tgt=<file> [options]

Options:
    -train_src=<file>         File of training source sentences
    -train_tgt=<file>         File of training target sentences
    -size=<int>               vocab size [default: 50000]
    -freq_cutoff=<int>        frequency cutoff [default: 2]
    -vocab_file=<file>        File of saving vocabulary
```

## Training
```
python run.py -mode=train -train_src=<file> -train_tgt=<file> -dev_src=<file> -dev_tgt=<file> -vocab=<file> [options]

Options:
    -mode=<src>                             train or test model [default: train]
    -cuda=<int>                             use which gpu, negative integer for cpu [default: 0]
    -train_src=<file>                       train source file
    -train_tgt=<file>                       train target file
    ...
# for full options, refer to run.py
```

## Evaluation
```
python run.py -mode=test  -test_src=<file>  [options]
Options:
    same as above
```
