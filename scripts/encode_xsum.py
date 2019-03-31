import argparse
import json
import os
from newsroom import jsonl
from tqdm import tqdm
from text_utils import TextEncoder

def load_summary(file_name):
    summary = []
    text = []
    is_summary = False
    is_text = False
    with open(file_name, encoding="latin-1") as f:
        for line in f:
            line = line.strip()
            if line[:4] == "[SN]" and line[-4:] == "[SN]":
                if line[4:-4] == "FIRST-SENTENCE":
                    is_summary = True
                elif line[4:-4] == "RESTBODY":
                    is_text = True
                    is_summary = False
            elif is_summary:
                if line:
                    summary.append(line)
            elif is_text:
                if line:
                    text.append(line)
    if len(summary) != 1 or not text:
        return {"summary": "", "text": ""}
    return {"summary": " ".join(summary), "text": " ".join(text)}

def load_splits(splits_file):
    with open(args.splits_file) as f:
        splits = json.load(f)
    train_split = set(splits["train"])
    val_split = set(splits["validation"])
    test_split = set(splits["test"])
    return train_split, val_split, test_split

def encode_line(line, encoder):
    encoding = encoder.encode([line])
    return encoding[0]

def main(args):
    text_encoder = TextEncoder(args.encoder_path, args.bpe_path)
    train_split, val_split, test_split = load_splits(args.splits_file)
    summaries = os.listdir(args.summary_dir)

    num_summaries = 0
    train_data, val_data, test_data = [], [], []
    for file_name in tqdm(summaries):
        summary_data = load_summary(os.path.join(args.summary_dir, file_name))
        if len(summary_data["summary"]) == 0 or len(summary_data["text"]) == 0:
            continue
        summary_data["summary"] = encode_line(summary_data["summary"], text_encoder)
        summary_data["text"] = encode_line(summary_data["text"], text_encoder)
        file_id = file_name.split(".")[0]
        if file_id in train_split:
            train_data.append(summary_data)
            num_summaries += 1
        elif file_id in val_split:
            val_data.append(summary_data)
            num_summaries += 1
        elif file_id in test_split:
            test_data.append(summary_data)
            num_summaries += 1

    with jsonl.open(args.train_file, gzip=True) as train_file:
        train_file.write(train_data)
    with jsonl.open(args.val_file, gzip=True) as val_file:
        val_file.write(val_data)
    with jsonl.open(args.test_file, gzip=True) as test_file:
        test_file.write(test_data)
    print("Number of successful conversions: {}".format(num_summaries))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--summary_dir', type=str, required=True)
    parser.add_argument('--splits_file', type=str, required=True)
    parser.add_argument('--train_file', type=str, required=True)
    parser.add_argument('--val_file', type=str, required=True)
    parser.add_argument('--test_file', type=str, required=True)
    parser.add_argument('--encoder_path', type=str, default='model/encoder_bpe_40000.json')
    parser.add_argument('--bpe_path', type=str, default='model/vocab_40000.bpe')
    args = parser.parse_args()

    main(args)
