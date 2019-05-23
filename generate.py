import argparse
import os
import random

import numpy as np
import rouge
import torch
from tqdm import tqdm


from data_loader import get_loader
from model_pytorch import LMModel, load_openai_pretrained_model
from parallel import DataParallelModel
from text_utils import TextEncoder

def get_rouge_scores(scores, index):
    rouge1 = [scores['rouge-1'][index]['f'][0], scores['rouge-1'][index]['p'][0], scores['rouge-1'][index]['r'][0]]
    rouge2 = [scores['rouge-2'][index]['f'][0], scores['rouge-2'][index]['p'][0], scores['rouge-2'][index]['r'][0]]
    rougel = [scores['rouge-l'][index]['f'][0], scores['rouge-l'][index]['p'][0], scores['rouge-l'][index]['r'][0]]
    return [rouge1, rouge2, rougel]

def generate_outputs(model, pad_output, mask_output, text_encoder, device, beam, gen_len, k, decoding_strategy, min_len=None):
    src_strs, tgt_strs, gen_strs = [], [], []
    mask = mask_output
    outputs = model(pad_output, mask_output, text_encoder, device, beam=beam, gen_len=gen_len, k=k, decoding_strategy=decoding_strategy, generate=True, min_len=min_len)
    for generated_toks, input_toks, target_toks in outputs:
        for idx in range(generated_toks.size(0)):
            src_str = toks_to_str(input_toks[idx], text_encoder, is_input=True, mask=mask[idx])
            src_strs.append(src_str)
            tgt_str = toks_to_str(target_toks[idx], text_encoder)
            tgt_strs.append(tgt_str)
            gen_str = toks_to_str(generated_toks[idx], text_encoder)
            gen_strs.append(gen_str)
    return src_strs, tgt_strs, gen_strs

def toks_to_str(toks, text_encoder, is_input=False, mask=None):
    str_rep = ''
    end_tok = text_encoder.encoder['_delimiter_'] if is_input else text_encoder.encoder['_classify_']
    for token in toks:
        if token.item() == end_tok:# or x.item() == end_idx:
            break
        elif token.item() in text_encoder.decoder:
            str_rep += text_encoder.decoder[token.item()].replace('</w>', ' ').replace('\n', '')
        else:
            str_rep += 'unk '
    # This makes sure rouge scorers doesn't complain about no sentences
    if not str_rep:
        str_rep = "unk."
    elif "." not in str_rep:
        str_rep += "."
    return str_rep

def init(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

def main(args):
    init(args)

    # Constants
    n_ctx = args.n_ctx
    data_dir = args.data_dir

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    print("device", device, "n_gpu", n_gpu)

    text_encoder = TextEncoder(args.encoder_path, args.bpe_path)
    encoder = text_encoder.encoder
    n_vocab = len(text_encoder.encoder)
    text_encoder.decoder[len(encoder)] = '_start_'
    encoder['_start_'] = len(encoder)
    text_encoder.decoder[len(encoder)] = '_delimiter_'
    encoder['_delimiter_'] = len(encoder)
    text_encoder.decoder[len(encoder)] = '_classify_'
    encoder['_classify_'] = len(encoder)

    n_special = 3   # XD: useless for language modeling task
    vocab = n_vocab + n_special + n_ctx

    lm_model = LMModel(args, vocab, n_ctx, return_probs=True, doc_embed=args.doc_model)
    load_openai_pretrained_model(lm_model.transformer, n_ctx=n_ctx, n_special=n_special)
    if args.checkpoint != "none":
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        state_dict = checkpoint["state_dict"]
        for key in list(state_dict.keys()):
            state_dict[key[7:]] = state_dict[key]
            del state_dict[key]
        pos_emb_mask = torch.zeros(1, 1, vocab)
        pos_emb_mask[:, :, -n_ctx] = -1e12
        state_dict['pos_emb_mask'] = pos_emb_mask
        lm_model.load_state_dict(state_dict)
    lm_model.to(device)
    lm_model = DataParallelModel(lm_model)

    train_bar = get_loader(os.path.join(data_dir, "val_encoded.jsonl"), n_gpu, encoder, num_workers=1, shuffle=True, max_size=args.n_iter)
    srcs, hyps, refs = [], [], []
    with torch.no_grad():
        lm_model.eval()
        for i, (pad_output, mask_output) in enumerate(tqdm(train_bar), 1):
            src_strs, tgt_strs, gen_strs = generate_outputs(lm_model, pad_output, mask_output, text_encoder, device, args.beam, args.gen_len, args.k, args.decoding_strategy)
            srcs.extend(src_strs)
            hyps.extend(gen_strs)
            refs.extend(tgt_strs)

    scorer = rouge.Rouge(metrics=['rouge-n', 'rouge-l'],
                         max_n=4,
                         limit_length=True,
                         length_limit=110,
                         length_limit_type='words',
                         apply_avg=False,
                         apply_best=False,
                         alpha=0.5, # Default F1_score
                         weight_factor=1.2,
                         stemming=True)
    scores = scorer.get_scores(hyps, refs)
    total_rouge1, total_rouge2, total_rougel = [0, 0, 0], [0, 0, 0], [0, 0, 0]
    for i in range(len(hyps)):
        print("*" * 50)
        print("Source: {}".format(srcs[i]))
        print('Hypothesis: {}'.format(hyps[i]))
        print("Reference: {}".format(refs[i]))
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
        print("\tRouge-1: (f: {:.5f}, p: {:.5f}, r: {:.5f})\n\tRouge-2: (f: {:.5f}, p: {:.5f}, r: {:.5f})\n\tRouge-l: (f: {:.5f}, p: {:.5f}, r: {:.5f})".format(*rouge1, *rouge2, *rougel))
    print("*" * 50 + "\n")
    print("*" * 50 + "\n")
    print("Averages")
    total_rouge1[0] /= len(hyps)
    total_rouge1[1] /= len(hyps)
    total_rouge1[2] /= len(hyps)
    total_rouge2[0] /= len(hyps)
    total_rouge2[1] /= len(hyps)
    total_rouge2[2] /= len(hyps)
    total_rougel[0] /= len(hyps)
    total_rougel[1] /= len(hyps)
    total_rougel[2] /= len(hyps)
    print("\tRouge-1: (f: {:.5f}, p: {:.5f}, r: {:.5f})\n\tRouge-2: (f: {:.5f}, p: {:.5f}, r: {:.5f})\n\tRouge-l: (f: {:.5f}, p: {:.5f}, r: {:.5f})".format(*total_rouge1, *total_rouge2, *total_rougel))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Standard
    parser.add_argument('--data_dir', type=str, default='data/')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_iter', type=int, default=1)
    parser.add_argument('--n_ctx', type=int, default=512)
    parser.add_argument('--n_embd', type=int, default=768)
    parser.add_argument('--n_head', type=int, default=12)
    parser.add_argument('--n_layer', type=int, default=12)
    parser.add_argument('--embd_pdrop', type=float, default=0.1)
    parser.add_argument('--attn_pdrop', type=float, default=0.1)
    parser.add_argument('--resid_pdrop', type=float, default=0.1)
    parser.add_argument('--clf_pdrop', type=float, default=0.1)
    parser.add_argument('--afn', type=str, default='gelu')
    parser.add_argument('--encoder_path', type=str, default='src/model/encoder_bpe_40000.json')
    parser.add_argument('--bpe_path', type=str, default='src/model/vocab_40000.bpe')
    parser.add_argument('--checkpoint', type=str, default="none")
    # Custom
    parser.add_argument('--gen_len', type=int, default=110,
                        help='Length of the generation')
    parser.add_argument('--k', type=int, default=10,
                        help='How many tokens to sample for various decoding strategies')
    parser.add_argument('--inits', type=str, default='init.txt',
                        help='Text file containing prefixes to continue')
    parser.add_argument('--decoding_strategy', type=int, default=0,
                        help='Which decoding strategy to use, described in the comments')
    parser.add_argument('--beam', type=int, default=0,
                        help='If this is 0, decoding_strategy will be used, if this is greater than 0 beam search will be used with the specified beam size')
    parser.add_argument('--doc_model', action='store_true',
                        help='Set to use the document embedding model')
    parser.add_argument('--min_len', type=int, default=None
                        help='Set to use the document embedding model')

    args = parser.parse_args()
    print(args)
    main(args)
