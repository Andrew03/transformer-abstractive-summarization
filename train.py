import argparse
import os
import random

import numpy as np
import rouge
import torch
from torch import nn
from tqdm import tqdm

from data_loader import get_loader
from generate import generate_outputs
from logger import Logger
from loss import LMLoss, SummaryLoss
from model_pytorch import LMModel, load_openai_pretrained_model
from opt import OpenAIAdam
from parallel import DataParallelModel, DataParallelCriterion
from text_utils import TextEncoder

def load_checkpoint(checkpoint_file, model, model_opt, vocab, n_ctx):
    """
    Loads a checkpoint including model state and running loss for continued training
    """
    if checkpoint_file is not None:
        checkpoint = torch.load(checkpoint_file)
        state_dict = checkpoint["state_dict"]
        start_iter = checkpoint['iter']
        running_loss = checkpoint['running_loss']
        opt_state_dict = checkpoint['optimizer']
        model_opt.load_state_dict(opt_state_dict)
        for state in model_opt.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.cuda()
        for key in list(state_dict.keys()):
            state_dict[key[7:]] = state_dict[key]
            del state_dict[key]
        pos_emb_mask = torch.zeros(1, 1, vocab)
        pos_emb_mask[:, :, -n_ctx] = -1e12
        model.load_state_dict(state_dict)
    else:
        start_iter = 1
        running_loss = 0
    return start_iter, running_loss

def get_average_scores(hyps, refs):
    rouge_scorer = rouge.Rouge(metrics=['rouge-n', 'rouge-l'],
                               max_n=4,
                               limit_length=True,
                               length_limit=110,
                               length_limit_type='words',
                               apply_avg=False,
                               apply_best=False,
                               alpha=0.5, # Default F1_score
                               weight_factor=1.2,
                               stemming=True)

    averaged_scores = {'rouge-1': {'f': 0, 'p': 0, 'r': 0},
                       'rouge-2': {'f': 0, 'p': 0, 'r': 0},
                       'rouge-l': {'f': 0, 'p': 0, 'r': 0}}
    scores = rouge_scorer.get_scores(hyps, refs)
    for metric in averaged_scores.keys():
        for values in scores[metric]:
            for sub_metric in averaged_scores[metric]:
                averaged_scores[metric][sub_metric] += values[sub_metric][0]
    for key in averaged_scores.keys():
        for sub_key in averaged_scores[key].keys():
            averaged_scores[key][sub_key] /= len(hyps)
    return averaged_scores

def run_batch(model, pad_seq, mask_seq, device, compute_loss_fct):
    pad_seq = pad_seq.to(device)
    mask_seq = mask_seq.to(device)
    lm_logits = model(pad_seq, mask_seq)
    loss = compute_loss_fct(lm_logits, pad_seq, mask_seq).mean()
    return loss

def save_checkpoint(num_updates, iter_num, running_loss, model_state_dict, optimizer_state_dict, save_dir):
    torch.save({
        "iter": iter_num,
        "running_loss": running_loss,
        "state_dict": model_state_dict,
        "optimizer": optimizer_state_dict
    }, os.path.join(save_dir, "checkpoint_{0:05d}.pt".format(num_updates)))

def evaluate(val_loader, train_log_interval, model, text_encoder, device, beam, gen_len, k, decoding_strategy, compute_loss_fct):
    hyps, refs = [], []
    val_loss = 0
    for j, (pad_seq, mask_seq) in enumerate(val_loader):
        with torch.no_grad():
            if j == train_log_interval:
                break
            if j <= 20:
                model.eval()
                # Generating outputs for evaluation
                src_strs, new_refs, new_hyps = generate_outputs(model, pad_seq, mask_seq, text_encoder, device, beam, gen_len, k, decoding_strategy)
                hyps.extend(new_hyps)
                refs.extend(new_refs)
            # Calculating loss
            val_loss += run_batch(model, pad_seq, mask_seq, device, compute_loss_fct).item()
    scores = get_average_scores(hyps, refs)
    return val_loss, scores

def run_epoch(start_iter, running_loss, model, compute_loss_fct, model_opt, train_loader, val_loader, train_log_interval, val_log_interval, device, beam, gen_len, k, decoding_strategy, accum_iter, desc_str, save_dir, logger, text_encoder, show_progress=False, summary_loss=None):
    if show_progress:
        train_bar = tqdm(iterable=train_loader, desc=desc_str)
    else:
        train_bar = train_loader

    for i, (pad_seq, mask_seq) in enumerate(train_bar, start_iter):
        num_updates = i // accum_iter
        model.train()
        loss = run_batch(model, pad_seq, mask_seq, device, compute_loss_fct)
        torch.cuda.empty_cache()
        loss.backward()
        running_loss += loss.item()

        if show_progress:
            train_bar.set_postfix(loss=running_loss / ((train_log_interval * accum_iter) if num_updates % train_log_interval == 0 and num_updates != 0 else i % (train_log_interval * accum_iter)))

        if i % accum_iter == 0:
            model_opt.step()
            model_opt.zero_grad()

        if num_updates % train_log_interval == 0 and i % accum_iter == 0:
            logger.scalar_summary("Training/Loss", running_loss / (train_log_interval * accum_iter), num_updates)
            running_loss = 0

        if num_updates % val_log_interval == 0 and i % accum_iter == 0:
            val_loss, scores = evaluate(val_loader, train_log_interval, model, text_encoder, device, beam, gen_len, k, decoding_strategy, summary_loss if summary_loss else compute_loss_fct)
            for key, value in scores.items():
                for key2, value2 in value.items():
                    logger.scalar_summary("{}/{}".format(key, key2), value2, num_updates)
            logger.scalar_summary("Validation/Loss", val_loss / train_log_interval, num_updates)
            torch.cuda.empty_cache()

        # Saving the model
        if num_updates % val_log_interval == 0 and i % accum_iter == 0:
            save_checkpoint(num_updates, i + 1, running_loss, model.state_dict(), model_opt.state_dict(), save_dir)
    save_checkpoint(num_updates, i + 1, running_loss, model.state_dict(), model_opt.state_dict(), save_dir)
    return i + 1, running_loss

def init(args):
    print("Creating directories")
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, args.experiment_name), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, args.experiment_name), exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

def main(args):
    init(args)

    # Constants
    n_ctx = args.n_ctx
    save_dir = os.path.join(args.output_dir, args.experiment_name, "checkpoints")
    desc = args.desc
    data_dir = args.data_dir
    log_dir = os.path.join(args.output_dir, args.experiment_name, "logs")
    train_log_interval = args.train_log_interval
    val_log_interval = args.val_log_interval
    beam = args.beam
    gen_len = args.gen_len
    k = args.k
    decoding_strategy = args.decoding_strategy
    accum_iter = args.accum_iter

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    print("device", device, "n_gpu", n_gpu)
    logger = Logger(log_dir)

    text_encoder = TextEncoder(args.encoder_path, args.vocab_path)
    encoder = text_encoder.encoder
    n_vocab = len(text_encoder.encoder)
    encoder['_start_'] = len(encoder)
    encoder['_delimiter_'] = len(encoder)
    encoder['_classify_'] = len(encoder)
    clf_token = encoder['_classify_']
    n_special = 3

    print("Loading dataset...")
    train_loader = get_loader(os.path.join(data_dir, "train_encoded.jsonl"), args.batch_size, encoder, num_workers=3, shuffle=True)
    val_loader = get_loader(os.path.join(data_dir, "val_encoded.jsonl"), n_gpu, encoder, num_workers=0, shuffle=False, max_size=args.num_val_examples)
    print("Train length: {}, Validation length: {}".format(len(train_loader), len(val_loader)))

    vocab = n_vocab + n_special + n_ctx
    n_updates_total = (len(train_loader) // args.accum_iter) * (args.num_epochs_dat + args.num_epochs_ft)

    dh_model = LMModel(args, vocab=vocab, n_ctx=n_ctx, doc_embed=args.doc_model)

    criterion = nn.CrossEntropyLoss(reduction="none")
    model_opt = OpenAIAdam(dh_model.parameters(),
                           lr=args.lr,
                           schedule=args.lr_schedule,
                           warmup=args.lr_warmup,
                           t_total=n_updates_total,
                           b1=args.b1,
                           b2=args.b2,
                           e=args.e,
                           l2=args.l2,
                           vector_l2=args.vector_l2,
                           max_grad_norm=args.max_grad_norm)

    lm_loss = LMLoss(criterion)
    summary_loss = SummaryLoss(criterion)

    print("Loading Model")
    if args.use_pretrain:
        load_openai_pretrained_model(dh_model.transformer, n_ctx=n_ctx, n_special=n_special, path="./model/", path_names="./")
    start_iter, running_loss = load_checkpoint(args.checkpoint, dh_model, model_opt, vocab, n_ctx)

    dh_model.to(device)
    dh_model = DataParallelModel(dh_model)
    lm_loss = DataParallelCriterion(lm_loss)
    summary_loss = DataParallelCriterion(summary_loss)

    for i in range(args.num_epochs_dat):
        start_iter, running_loss = run_epoch(start_iter, running_loss, dh_model, lm_loss, model_opt, train_loader, val_loader, train_log_interval, val_log_interval, device, beam, gen_len, k, decoding_strategy, accum_iter, "DAT Training Epoch [{}/{}]".format(i + 1, args.num_epochs_dat), save_dir, logger, text_encoder, show_progress=args.show_progress, summary_loss=summary_loss)
    for i in range(args.num_epochs_ft):
        start_iter, running_loss = run_epoch(start_iter, running_loss, dh_model, summary_loss, model_opt, train_loader, val_loader, train_log_interval, val_log_interval, device, beam, gen_len, k, decoding_strategy, accum_iter, "FT Training Epoch [{}/{}]".format(i + 1, args.num_epochs_ft), save_dir, logger, text_encoder, show_progress=args.show_progress)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--desc', type=str, help="Description")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_epochs_dat', type=int, default=0)
    parser.add_argument('--num_epochs_ft', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_grad_norm', type=int, default=1)
    parser.add_argument('--lr', type=float, default=6.25e-5)
    parser.add_argument('--lr_warmup', type=float, default=0.002)
    parser.add_argument('--n_ctx', type=int, default=512)
    parser.add_argument('--n_embd', type=int, default=768)
    parser.add_argument('--n_head', type=int, default=12)
    parser.add_argument('--n_layer', type=int, default=12)
    parser.add_argument('--embd_pdrop', type=float, default=0.1)
    parser.add_argument('--attn_pdrop', type=float, default=0.1)
    parser.add_argument('--resid_pdrop', type=float, default=0.1)
    parser.add_argument('--clf_pdrop', type=float, default=0.1)
    parser.add_argument('--l2', type=float, default=0.01)
    parser.add_argument('--vector_l2', action='store_true')
    parser.add_argument('--opt', type=str, default='adam')
    parser.add_argument('--afn', type=str, default='gelu')
    parser.add_argument('--lr_schedule', type=str, default='warmup_linear')
    parser.add_argument('--encoder_path', type=str, default='model/encoder_bpe_40000.json')
    parser.add_argument('--vocab_path', type=str, default='model/vocab_40000.bpe')
    parser.add_argument('--n_transfer', type=int, default=12)
    parser.add_argument('--lm_coef', type=float, default=0.5)
    parser.add_argument('--b1', type=float, default=0.9)
    parser.add_argument('--b2', type=float, default=0.999)
    parser.add_argument('--e', type=float, default=1e-8)
    # Custom
    parser.add_argument('--output_dir', type=str, default="output")
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--experiment_name', type=str, required=True)
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--train_log_interval', type=int, default=100)
    parser.add_argument('--val_log_interval', type=int, default=2000)
    parser.add_argument('--num_val_examples', type=int, default=500)
    parser.add_argument('--beam', type=int, default=3)
    parser.add_argument('--gen_len', type=int, default=110)
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--decoding_strategy', type=int, default=0)
    parser.add_argument('--accum_iter', type=int, default=2)
    parser.add_argument('--show_progress', action='store_true')
    parser.add_argument('--doc_model', action='store_true')
    parser.add_argument('--use_pretrain', action='store_true')
    args = parser.parse_args()
    print(args)
    main(args)
