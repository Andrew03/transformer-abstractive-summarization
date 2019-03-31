import argparse
from newsroom import jsonl
from tqdm import tqdm
from text_utils import TextEncoder

def encode_line(line, encoder):
    encoding = encoder.encode([line])
    return encoding[0]

def main(args):
    text_encoder = TextEncoder(args.encoder_path, args.bpe_path)
    num_summaries = 0
    out_data = []
    with jsonl.open(args.in_file, gzip=True) as in_file:
        data = in_file.read()
        for entry in tqdm(data):
            if entry["summary"] is None or entry["text"] is None:
                continue
            entry["summary"] = encode_line(entry["summary"], text_encoder)
            entry["text"] = encode_line(entry["text"], text_encoder)
            num_summaries += 1
            out_data.append(entry)
    with jsonl.open(args.out_file, gzip=True) as out_file:
        out_file.write(out_data)
    print("Number of successful conversions: {}".format(num_summaries))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_file', type=str, required=True)
    parser.add_argument('--out_file', type=str, required=True)
    parser.add_argument('--encoder_path', type=str, default='model/encoder_bpe_40000.json')
    parser.add_argument('--bpe_path', type=str, default='model/vocab_40000.bpe')
    args = parser.parse_args()

    main(args)
