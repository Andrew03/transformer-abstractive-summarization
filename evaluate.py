import argparse
import os
import random

import numpy as np
import rouge
import torch
from tqdm import tqdm

from data_loader import get_loader
from generate import generate_outputs, get_rouge_scores
from model_pytorch import LMModel, load_openai_pretrained_model
from parallel import DataParallelModel
from text_utils import TextEncoder


def evaluate_model(model, val_loader, text_encoder, device, beam, gen_len, k, decoding_strategy, save_file="none", args=None):
    srcs, hyps, refs = [], [], []
    scorer = rouge.Rouge(metrics=['rouge-n', 'rouge-l'],
                         max_n=4,
                         limit_length=True,
                         length_limit=110,
                         length_limit_type='words',
                         apply_avg=False,
                         apply_best=False,
                         alpha=0.5,
                         weight_factor=1.2,
                         stemming=True)

    model.eval()
    for pad_seq, mask_seq in tqdm(val_loader):
        with torch.no_grad():
            # Generating outputs for evaluation
            src_strs, tgt_strs, gen_strs = generate_outputs(model, pad_seq, mask_seq, text_encoder, device, beam, gen_len, k, decoding_strategy)
            srcs.extend(src_strs)
            hyps.extend(gen_strs)
            refs.extend(tgt_strs)

    total_rouge1, total_rouge2, total_rougel = [0, 0, 0], [0, 0, 0], [0, 0, 0]
    if save_file != "none":
        scores = scorer.get_scores(hyps, refs)
        with open(save_file, "w") as f:
            f.write("{}\n".format(args))
            for i in range(len(srcs)):
                f.write("-"*50 + "\n")
                f.write("Src: {}\n".format(srcs[i]))
                f.write("Tgt: {}\n".format(refs[i]))
                f.write("Gen: {}\n".format(hyps[i]))
                rouge1, rouge2, rougel = get_rouge_scores(scores, i)
                total_rouge1[0] += rouge1[0]
                total_rouge1[1] += rouge1[1]
                total_rouge1[2] += rouge1[2]
                total_rouge2[0] += rouge2[0]
                total_rouge2[1] += rouge2[1]
                total_rouge2[2] += rouge2[2]
                total_rougel[0] += rougel[0]
                total_rougel[1] += rougel[1]
                total_rougel[2] += rougel[2]
                f.write("\tRouge-1: (f: {:.5f}, p: {:.5f}, r: {:.5f})\n\tRouge-2: (f: {:.5f}, p: {:.5f}, r: {:.5f})\n\tRouge-L: (f: {:.5f}, p: {:.5f}, r: {:.5f})\n".format(*rouge1, *rouge2, *rougel))
            f.write("-"*50 + "\n")
            f.write("-"*50 + "\n")
            f.write("Average:")
            total_rouge1[0] /= len(hyps)
            total_rouge1[1] /= len(hyps)
            total_rouge1[2] /= len(hyps)
            total_rouge2[0] /= len(hyps)
            total_rouge2[1] /= len(hyps)
            total_rouge2[2] /= len(hyps)
            total_rougel[0] /= len(hyps)
            total_rougel[1] /= len(hyps)
            total_rougel[2] /= len(hyps)
            f.write("\tRouge-1: (f: {:.5f}, p: {:.5f}, r: {:.5f})\n\tRouge-2: (f: {:.5f}, p: {:.5f}, r: {:.5f})\n\tRouge-L: (f: {:.5f}, p: {:.5f}, r: {:.5f})\n".format(*total_rouge1, *total_rouge2, *total_rougel))
    else:
        scores = scorer.get_scores(hyps, refs)
        for i in range(len(srcs)):
            rouge1, rouge2, rougel = get_rouge_scores(scores, i)
            total_rouge1[0] += rouge1[0]
            total_rouge1[1] += rouge1[1]
            total_rouge1[2] += rouge1[2]
            total_rouge2[0] += rouge2[0]
            total_rouge2[1] += rouge2[1]
            total_rouge2[2] += rouge2[2]
            total_rougel[0] += rougel[0]
            total_rougel[1] += rougel[1]
            total_rougel[2] += rougel[2]
        total_rouge1[0] /= len(hyps)
        total_rouge1[1] /= len(hyps)
        total_rouge1[2] /= len(hyps)
        total_rouge2[0] /= len(hyps)
        total_rouge2[1] /= len(hyps)
        total_rouge2[2] /= len(hyps)
        total_rougel[0] /= len(hyps)
        total_rougel[1] /= len(hyps)
        total_rougel[2] /= len(hyps)
    print("\tRouge-1: (f: {:.5f}, p: {:.5f}, r: {:.5f})\n\tRouge-2: (f: {:.5f}, p: {:.5f}, r: {:.5f})\n\tRouge-L: (f: {:.5f}, p: {:.5f}, r: {:.5f})\n".format(*total_rouge1, *total_rouge2, *total_rougel))

def main(args):
    # Constants
    n_ctx = args.n_ctx
    desc = args.desc
    data_dir = args.data_dir

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
    val_loader = get_loader(os.path.join(data_dir, "val_encoded.jsonl"), args.n_batch, encoder, num_workers=6, shuffle=False)
    test_loader = get_loader(os.path.join(data_dir, "test_encoded.jsonl"), args.n_batch, encoder, num_workers=6, shuffle=False)

    vocab = n_vocab + n_special + n_ctx
    dh_model = LMModel(args, vocab=vocab, n_ctx=n_ctx, doc_embed=args.doc_model)

    print("Loading model...")
    load_openai_pretrained_model(dh_model.transformer, n_ctx=n_ctx, n_special=n_special, path="src/model/", path_names="src/")
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

    evaluate_model(dh_model, test_loader if args.use_test else val_loader, text_encoder, device, args.beam, args.gen_len, args.k, args.decoding_strategy, args.save_file, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--desc', type=str, help="Description")
    parser.add_argument('--data_dir', type=str, default='data/')
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
    parser.add_argument('--encoder_path', type=str, default='src/model/encoder_bpe_40000.json')
    parser.add_argument('--bpe_path', type=str, default='src/model/vocab_40000.bpe')
    parser.add_argument('--checkpoint', type=str, default="none")
    parser.add_argument('--beam', type=int, default=0)
    parser.add_argument('--gen_len', type=int, default=110)
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--decoding_strategy', type=int, default=0)
    parser.add_argument('--save_file', type=str, default="none")
    parser.add_argument('--doc_model', action='store_true')
    parser.add_argument('--use_test', action='store_true')

    args = parser.parse_args()
    print(args)

    os.makedirs(args.output_dir, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    main(args)
