import argparse
from text_utils import TextEncoder
from newsroom import jsonl

def main(args):
    text_encoder = TextEncoder(args.encoder_path, args.bpe_path)
    with jsonl.open(args.original_file, gzip = True) as test_file:
        data = test_file.read()

    with jsonl.open(args.out_file, gzip=True) as out_file:
        out_file.write(data[-args.n:])

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Creates a data file using the last n examples from the original file. Used to cover the fact that the model drops the last few results")
    parser.add_argument("--original_file", type=str, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--out_file", type=str, required=True)
    parser.add_argument('--encoder_path', type=str, default='model/encoder_bpe_40000.json')
    parser.add_argument('--bpe_path', type=str, default='model/vocab_40000.bpe')
    args = parser.parse_args()
    main(args)
