# transformer-abstractive-summarization

Code for the paper "Deep Transformers for Abstractive Summarization"

## Requirements

To run the training script in [train.py](train.py) you will need in addition:
- PyTorch (version >=0.4)
- tqdm
- py-rouge
- [newsroom](https://github.com/clic-lab/newsroom)
- tensorflow (cpu version is ok)
- spacy (and 'en' model)

You can download the weights of the OpenAI pre-trained version by cloning [Alec Radford's repo](https://github.com/openai/finetune-transformer-lm) and placing the `model` folder containing the pre-trained weights in the present repo.

In order to run this code, you will need to pre-process the datasets using bpe through the scripts provided in [scripts](scripts)
## Dataset Preprocessing
The training and evaluation scripts expect 3 total output files: `train_encoded.jsonl`, `val_encoded.jsonl`, and `test_encoded.jsonl`

### CNN/Daily Mail
The data and splits used in the paper can be downloaded from [OpenNMT](http://opennmt.net/OpenNMT-py/Summarization.html). 
To process the data, run the following command:
```
python scripts/encode_cnndm.py --src_file {source file} --tgt_file {target file} --out_file {output file}
```

### XSum
The data and splits used in the paper can be downloaded from [XSum](https://github.com/EdinburghNLP/XSum/tree/master/XSum-Dataset). 
To process the data, run the following command:
```
python scripts/encode_xsum.py --summary_dir {summary directory} --splits_file {split file} --train_file {train file} --val_file {val file} --test_file {test_file}
```

### Newsroom
The data and splits used in the paper can be downloaded from [Newsroom](https://github.com/clic-lab/newsroom). 
To process the data, run the following command:
```
python scripts/encode_newsroom.py --in_file {input split file} --out_file {output file}
```

## Training
To train a model, run the following command:
```
python train.py \
  --data_dir {directory containing encoded data} \
  --output_dir {name of folder to save data in} \
  --experiment_name {name of experiment to save data with} \
  --show_progress \
  --doc_model \
  --num_epochs_dat 10 \
  --num_epochs_ft 10 \
  --use_pretrain
```
to train the pre-trained document embedding model over `dataset` for 10 epochs using domain adaptive training, and 10 epochs using fine tuning. Training is currently optimized for multi-gpu usage, and may not work for single gpu machines.

## Evaluation
To evaluate a model, run the following command:
```
python evaluate.py \
  --data_dir {directory containing encoded data} \
  --checkpoint {checkpoint to load model weights from} \
  --beam {beam size to do beam search with} \
  --doc_model \
  --use_test \
```
to evaluate the document embedding model on the test set. Evaluation is currently optimized for multi-gpu usage, and may not work for single gpu machines.

To evaluate a model on the newsroom validation set (use the previous command for all other evaluation), run the following command:
```
python evaluate.py \
  --data_dir {directory containing encoded data} \
  --checkpoint {checkpoint to load model weights from} \
  --beam {beam size to do beam search with} \
  --doc_model \
```
to evaluate the document embedding model on the validation set. Evaluation is currently optimized for multi-gpu usage, and may not work for single gpu machines.
