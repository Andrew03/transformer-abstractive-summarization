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
    with open(args.src_file) as src_file, open(args.tgt_file) as tgt_file:
        src_lines = src_file.readlines()
        tgt_lines = tgt_file.readlines()
        for i in tqdm(range(len(src_lines))):
            num_summaries += 1
            out_data.append({
                "summary": encode_line(tgt_lines[i].strip(), text_encoder), 
                "text": encode_line(src_lines[i].strip(), text_encoder)
            })
    with jsonl.open(args.out_file, gzip=True) as out_file:
        out_file.write(out_data)
    print("Number of successful conversions: {}".format(num_summaries))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_file', type=str, required=True)
    parser.add_argument('--tgt_file', type=str, required=True)
    parser.add_argument('--out_file', type=str, required=True)
    parser.add_argument('--encoder_path', type=str, default='model/encoder_bpe_40000.json')
    parser.add_argument('--bpe_path', type=str, default='model/vocab_40000.bpe')
    args = parser.parse_args()

    main(args)
