import argparse
import csv
import glob
import json
import os
import random
import re

from nltk.tokenize import sent_tokenize
import numpy as np
from pyrouge import Rouge155
import torch
import torch.nn as nn

from opt import OpenAIAdam
from text_utils import TextEncoder
from data_loader import get_loader
from tqdm import tqdm
from generate import generate_outputs
from model_pytorch import LMModel, load_openai_pretrained_model
from parallel import DataParallelModel

def clear_dirs(gen_dir, tgt_dir):
    for f in glob.glob("{}/*".format(tgt_dir)):
        os.remove(f)
    for f in glob.glob("{}/*".format(gen_dir)):
        os.remove(f)
    os.makedirs(tgt_dir, exist_ok=True)
    os.makedirs(gen_dir, exist_ok=True)

def format_text(text, max_len, stop_words=[]):
    text = "\n".join(sent_tokenize(text)).replace("<", "&lt").replace(">", "&gt")
    for stop_word in stop_words:
        text = text.replace(" {} ".format(stop_word), " ")
    if max_len is not None:
        text = " ".join(text.split(" ")[:max_len])
    return text

def evaluate_model(model, val_loader, text_encoder, device, beam, gen_len, k, decoding_strategy, save_file, gen_dir="gen", tgt_dir="tgt", max_len=110, stop_words=[], args=None):
    data = {"src": [], "gen": [], "tgt": []}
    srcs, hyps, refs = [], [], []

    model.eval()
    for pad_seq, mask_seq in tqdm(val_loader):
        with torch.no_grad():
            # Generating outputs for evaluation
            src_strs, tgt_strs, gen_strs = generate_outputs(model, pad_seq, mask_seq, text_encoder, device, beam, gen_len, k, decoding_strategy, min_len=args.min_len)
            data["src"].extend(src_strs)
            data["gen"].extend(gen_strs)
            data["tgt"].extend(tgt_strs)

    for i in range(len(data["src"])):
        with open(os.path.join(gen_dir, "gen.{}.txt".format(i)), "w") as gen_file:
            gen_file.write(format_text(data["gen"][i], max_len, stop_words))
        with open(os.path.join(tgt_dir, "tgt.{}.txt".format(i)), "w") as tgt_file:
            tgt_file.write(format_text(data["tgt"][i], max_len, stop_words))

    with open(save_file, "w") as f:
        json.dump(
            get_rouge_scores(gen_dir, tgt_dir),
            f,
            indent=4,
            sort_keys=True
        )

def get_rouge_scores(gen_dir, tgt_dir, gen_pattern='gen.(\d+).txt', tgt_pattern='tgt.#ID#.txt'):
    r = Rouge155()
    r.system_dir = gen_dir
    r.model_dir = tgt_dir
    r.system_filename_pattern = gen_pattern
    r.model_filename_pattern = tgt_pattern
    output = r.convert_and_evaluate()
    return r.output_to_dict(output)

def main(args):
    # Constants
    n_ctx = args.n_ctx
    desc = args.desc

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    print("device", device, "n_gpu", n_gpu)

    text_encoder = TextEncoder(args.encoder_path, args.bpe_path)
    encoder = text_encoder.encoder
    n_vocab = len(text_encoder.encoder)


    encoder['_start_'] = len(encoder)
    encoder['_delimiter_'] = len(encoder)
    encoder['_classify_'] = len(encoder)
    clf_token = encoder['_classify_']
    n_special = 3

    print("Loading dataset...")
    test_loader = get_loader(args.data_file, args.n_batch, encoder, num_workers=1, shuffle=False, subset=args.subset)

    vocab = n_vocab + n_special + n_ctx
    dh_model = LMModel(args, vocab=vocab, n_ctx=n_ctx, doc_embed=args.doc_model)

    print("Loading model...")
    load_openai_pretrained_model(dh_model.transformer, n_ctx=n_ctx, n_special=n_special, path="./model/", path_names="./")
    if args.checkpoint != "none":
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        state_dict = checkpoint["state_dict"]
        for key in list(state_dict.keys()):
            state_dict[key[7:]] = state_dict[key]
            del state_dict[key]
        pos_emb_mask = torch.zeros(1, 1, vocab)
        pos_emb_mask[:, :, -n_ctx] = -1e12
        state_dict['pos_emb_mask'] = pos_emb_mask
        dh_model.load_state_dict(state_dict)

    dh_model.to(device)
    dh_model = DataParallelModel(dh_model)

    stop_words = []
    if args.stop_words is not None:
        with open(args.stop_words) as f:
            for line in f:
                stop_words.append(line)
    evaluate_model(dh_model, test_loader, text_encoder, device, args.beam, args.gen_len, args.k, args.decoding_strategy, args.save_file, args.gen_dir, args.tgt_dir, args.max_len, stop_words, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--desc', type=str, help="Description")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_ctx', type=int, default=512)
    parser.add_argument('--n_embd', type=int, default=768)
    parser.add_argument('--n_head', type=int, default=12)
    parser.add_argument('--n_layer', type=int, default=12)
    parser.add_argument('--n_batch', type=int, default=1)
    parser.add_argument('--embd_pdrop', type=float, default=0.1)
    parser.add_argument('--attn_pdrop', type=float, default=0.1)
    parser.add_argument('--resid_pdrop', type=float, default=0.1)
    parser.add_argument('--clf_pdrop', type=float, default=0.1)
    parser.add_argument('--afn', type=str, default='gelu')
    parser.add_argument('--encoder_path', type=str, default='model/encoder_bpe_40000.json')
    parser.add_argument('--bpe_path', type=str, default='model/vocab_40000.bpe')
    parser.add_argument('--checkpoint', type=str, default="none")
    parser.add_argument('--data_file', type=str, required=True)
    parser.add_argument('--beam', type=int, default=0)
    parser.add_argument('--gen_len', type=int, default=110)
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--min_len', type=int, default=None)
    parser.add_argument('--decoding_strategy', type=int, default=0)
    parser.add_argument('--save_file', type=str, required=True)
    parser.add_argument('--doc_model', action='store_true')
    parser.add_argument('--gen_dir', type=str, default="gen")
    parser.add_argument('--tgt_dir', type=str, default="tgt")
    parser.add_argument('--max_len', type=int, default=110)
    parser.add_argument('--stop_words', type=str, default=None)
    parser.add_argument('--subset', type=str, default=None)

    args = parser.parse_args()
    print(args)


    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    clear_dirs(args.gen_dir, args.tgt_dir)
    main(args)
