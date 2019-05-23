import copy
import json
import math
import re

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


def swish(x):
    return x * torch.sigmoid(x)


ACT_FNS = {
    'relu': nn.ReLU,
    'swish': swish,
    'gelu': gelu
}


class LayerNorm(nn.Module):
    "Construct a layernorm module in the OpenAI style (epsilon inside the square root)."

    def __init__(self, n_state, e=1e-5):
        super(LayerNorm, self).__init__()
        self.g = nn.Parameter(torch.ones(n_state))
        self.b = nn.Parameter(torch.zeros(n_state))
        self.e = e

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.e)
        return self.g * x + self.b


class Conv1D(nn.Module):
    def __init__(self, nf, rf, nx):
        super(Conv1D, self).__init__()
        self.rf = rf
        self.nf = nf
        if rf == 1:  # faster 1x1 conv
            w = torch.empty(nx, nf)
            nn.init.normal_(w, std=0.02)
            self.w = Parameter(w)
            self.b = Parameter(torch.zeros(nf))
        else:  # was used to train LM
            raise NotImplementedError

    def forward(self, x):
        if self.rf == 1:
            size_out = x.size()[:-1] + (self.nf,)
            x = torch.addmm(self.b, x.view(-1, x.size(-1)), self.w)
            x = x.view(*size_out)
        else:
            raise NotImplementedError
        return x


class Attention(nn.Module):
    def __init__(self, nx, n_ctx, cfg, scale=False):
        super(Attention, self).__init__()
        n_state = nx  # in Attention: n_state=768 (nx=n_embd)
        # [switch nx => n_state from Block to Attention to keep identical to TF implem]
        assert n_state % cfg.n_head == 0
        self.register_buffer('b', torch.tril(torch.ones(625, 625)).view(1, 1, 625, 625))
        self.n_head = cfg.n_head
        self.split_size = n_state
        self.scale = scale
        self.c_attn = Conv1D(n_state * 3, 1, nx)
        self.c_proj = Conv1D(n_state, 1, nx)
        self.attn_dropout = nn.Dropout(cfg.attn_pdrop)
        self.resid_dropout = nn.Dropout(cfg.resid_pdrop)

    def _attn(self, q, k, v, mask):
        w = torch.matmul(q, k)
        if self.scale:
            w = w / math.sqrt(v.size(-1))
        # w = w * self.b + -1e9 * (1 - self.b)  # TF implem method: mask_attn_weights
        # XD: self.b may be larger than w, so we need to crop it
        b = self.b * mask.unsqueeze(1).unsqueeze(1).type_as(self.b)
        b = b[:, :, :w.size(-2), :w.size(-1)]
        w = w * b + -1e9 * (1 - b)

        w = nn.Softmax(dim=-1)(w)
        w = self.attn_dropout(w)
        return torch.matmul(w, v)

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)  # in Tensorflow implem: fct merge_states

    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)  # in Tensorflow implem: fct split_states
        if k:
            return x.permute(0, 2, 3, 1)
        else:
            return x.permute(0, 2, 1, 3)

    def forward(self, x, mask):
        x = self.c_attn(x)
        query, key, value = x.split(self.split_size, dim=2)
        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)
        a = self._attn(query, key, value, mask)
        a = self.merge_heads(a)
        a = self.c_proj(a)
        a = self.resid_dropout(a)
        return a


class MLP(nn.Module):
    def __init__(self, n_state, cfg):  # in MLP: n_state=3072 (4 * n_embd)
        super(MLP, self).__init__()
        nx = cfg.n_embd
        self.c_fc = Conv1D(n_state, 1, nx)
        self.c_proj = Conv1D(nx, 1, n_state)
        self.act = ACT_FNS[cfg.afn]
        self.dropout = nn.Dropout(cfg.resid_pdrop)

    def forward(self, x):
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        return self.dropout(h2)


class Block(nn.Module):
    def __init__(self, n_ctx, cfg, scale=False):
        super(Block, self).__init__()
        nx = cfg.n_embd
        self.attn = Attention(nx, n_ctx, cfg, scale)
        self.ln_1 = LayerNorm(nx)
        self.mlp = MLP(4 * nx, cfg)
        self.ln_2 = LayerNorm(nx)

    def forward(self, x, mask):
        a = self.attn(x, mask)
        n = self.ln_1(x + a)
        m = self.mlp(n)
        h = self.ln_2(n + m)
        return h


class TransformerModel(nn.Module):
    """ Transformer model """

    def __init__(self, cfg, vocab=40990, n_ctx=512, doc_embed=True):
        super(TransformerModel, self).__init__()
        self.vocab = vocab
        self.embed = nn.Embedding(vocab, cfg.n_embd)
        self.is_doc_embed = doc_embed
        if doc_embed:
            self.article_embed = nn.Embedding(2, cfg.n_embd)
            self.summary_embed = nn.Embedding(2, cfg.n_embd)
            nn.init.normal_(self.article_embed.weight, 0, 0.01)
            nn.init.normal_(self.summary_embed.weight, 0, 0.01)
        self.drop = nn.Dropout(cfg.embd_pdrop)
        block = Block(n_ctx, cfg, scale=True)
        self.h = nn.ModuleList([copy.deepcopy(block) for _ in range(cfg.n_layer)])

        nn.init.normal_(self.embed.weight, std=0.02)

    def forward(self, x, mask):
        x = x.view(-1, x.size(-2), x.size(-1))
        e = self.embed(x)
        # Add the position information to the input embeddings
        h = e.sum(dim=2)
        if self.is_doc_embed:
            doc_embed = torch.cat((self.article_embed(mask[:, :514]), self.summary_embed(mask[:, 514:])), dim=1)
            h += doc_embed[:, :h.size(1), :]
        for block in self.h:
            h = block(h, mask)
        return h


class LMHead(nn.Module):
    """ Language Model Head for the transformer """

    def __init__(self, model, cfg, trunc_and_reshape=True):
        super(LMHead, self).__init__()
        self.n_embd = cfg.n_embd
        embed_shape = model.embed.weight.shape
        self.decoder = nn.Linear(embed_shape[1], embed_shape[0], bias=False)
        self.decoder.weight = model.embed.weight # Tied weights
        self.trunc_and_reshape = trunc_and_reshape  # XD

    def forward(self, h):
        # Truncated Language modeling logits (we remove the last token)
        h_trunc = h[:, :-1].contiguous().view(-1, self.n_embd) \
            if self.trunc_and_reshape else h  # XD
        lm_logits = self.decoder(h_trunc)
        return lm_logits

# XD
class LMModel(nn.Module):
    """ Transformer with language model head only """
    def __init__(self, cfg, vocab=40990, n_ctx=512, return_probs=False, doc_embed=True):
        super(LMModel, self).__init__()
        self.transformer = TransformerModel(cfg, vocab=vocab, n_ctx=n_ctx, doc_embed=doc_embed)
        self.lm_head = LMHead(self.transformer, cfg, trunc_and_reshape=False)
        self.return_probs = return_probs

        #if self.return_probs:
        pos_emb_mask = torch.zeros(1, 1, vocab)
        pos_emb_mask[:, :, -n_ctx:] = -1e12
        self.register_buffer('pos_emb_mask', pos_emb_mask)

    def forward(self, pad_output, mask_output=None, text_encoder=None, device=None, beam=0, gen_len=110, k=0, decoding_strategy=0, log=True, generate=False, min_len=None):
        if generate:
            return self.generate(pad_output, mask_output, text_encoder, device, beam, gen_len, k, decoding_strategy, min_len=min_len)
        return self._forward(pad_output, mask_output, log)


    def _forward(self, x, mask_output, log=True, return_probs=False):
        h = self.transformer(x, mask_output)
        lm_logits = self.lm_head(h)
        if self.return_probs or return_probs:
            if log:
                lm_logits = F.log_softmax((lm_logits + self.pos_emb_mask), dim=-1)
            else:
                lm_logits = F.softmax((lm_logits + self.pos_emb_mask), dim=-1)
        return lm_logits

    def append_batch(self, X, next_idx):
        next_pos = X[:, -1:, 1] + 1
        next_x = torch.cat((next_idx, next_pos), -1).unsqueeze(1)
        return torch.cat((X, next_x), 1)


    def sample(self, pad_output, mask, classify_idx, text_encoder, gen_len=110, k=0, decoding_strategy=0, min_len=None):
        XMB = pad_output
        seen_trigrams = [{} for _ in range(XMB.size(0))]
        for _ in range(gen_len):
            lm_probs = self._forward(XMB, mask, return_probs=True, log=False)
            dist = lm_probs[:, -1, :].squeeze()
            if k == 0:
                next_idx = torch.multinomial(lm_probs[:, -1, :], 1)
            else:
                if decoding_strategy == 0:
                    # Sample from top k
                    values, indices = dist.topk(k)
                    next_idx = indices.gather(-1, torch.multinomial(values, 1))
                    if _ == 2:
                        for i in range(XMB.size(0)):
                            bigram = (XMB[i, -2, 0].item(), XMB[i, -1, 0].item())
                            seen_trigrams[i][bigram] = [next_idx[i].item()]
                    elif _ > 2:
                        for i in range(XMB.size(0)):
                            bigram = (XMB[i, -2, 0].item(), XMB[i, -1, 0].item())
                            if bigram in seen_trigrams[i]:
                                for value in seen_trigrams[i][bigram]:
                                    dist[i, value] = 0
                        values, indices = dist.topk(k)
                        next_idx = indices.gather(-1, torch.multinomial(values, 1))
                        for i in range(XMB.size(0)):
                            bigram = (XMB[i, -2, 0].item(), XMB[i, -1, 0].item())
                            if bigram not in seen_trigrams[i]:
                                seen_trigrams[i][bigram] = []
                            seen_trigrams[i][bigram].append(next_idx[i].item())
                else:
                    raise NotImplementedError
            XMB = self.append_batch(XMB, next_idx)
        return XMB[:, -gen_len:, 0]


    def beam_search(self, pad_output, mask, classify_idx, text_encoder, beam, gen_len=110, min_len=None):
        batch_size = pad_output.size(0)
        XMB = pad_output
        finished_beams = [[] for _ in range(batch_size)]

        """Initial run"""
        dist = self._forward(XMB, mask, log=True, return_probs=True)[:, -1, :]
        beam_lls, beam_toks = dist.topk(beam)
        beam_probs = beam_lls.view(-1, 1)
        beam_toks = beam_toks.view(-1, 1)
        XMB = XMB.repeat(1, beam, 1).view(-1, XMB.size(1), XMB.size(2))
        next_x = torch.cat((beam_toks, XMB[:, -1:, 1] + 1), -1).unsqueeze(1)
        XMB = torch.cat((XMB, next_x), 1)
        mask = mask.repeat(1, beam).view(-1, mask.size(1))

        finished_mask = beam_toks.eq(classify_idx)
        for i in range(finished_mask.size(0)):
            if finished_mask[i].item() == 1:
                finished_beams[i // beam].append((XMB[i, 1+512+1:, 0].cpu(), beam_probs[i].item() / XMB.size(1) - 513)) # 513 to include classify tok in error and avoid division by 0
                beam_probs[i] = -1e8

        for _ in range(gen_len - 1):
            top_values, top_beams = beam_probs.view(batch_size, -1).topk(beam)
            top_beams = (top_beams + torch.tensor([[i * beam_probs.size(0) / batch_size for j in range(beam)] for i in range(batch_size)]).type_as(top_beams)).view(-1)
            index = torch.cat([torch.ones(1, mask.size(1), XMB.size(2)).long() * torch.tensor([top_beams[i]]).long() for i in range(top_beams.size(0))], dim=0).type_as(XMB)
            XMB = torch.gather(XMB, 0, index[:, :XMB.size(1), :])
            mask = torch.gather(mask, 0, index[:, :, 0])
            beam_probs = torch.gather(beam_probs, 0, index[:, 0, 0].unsqueeze(1))
            if _ > 2:
                seen_hashes = torch.gather(seen_hashes, 0, index[:, :seen_hashes.size(1), 0])

            lm_probs = self._forward(XMB, mask, log=True, return_probs=True)[:, -1, :]
            if _ == 2:
                trigram = XMB[:, -3:, 0]
                trigram_hash = (2.0 * trigram[:, 0] + 3.3 * trigram[:, 1] + 7.8 * trigram[:, 2]).unsqueeze(1)
                seen_hashes = trigram_hash
            elif _ > 2:
                trigram = XMB[:, -3:, 0]
                trigram_hash = (2.0 * trigram[:, 0] + 3.3 * trigram[:, 1] + 7.8 * trigram[:, 2]).unsqueeze(1)
                trigram_mask = trigram_hash.eq(seen_hashes).sum(dim=1, keepdim=True).byte()
                lm_probs.masked_fill_(trigram_mask, -1e8)
                seen_hashes = torch.cat((seen_hashes, trigram_hash), dim=1)

            beam_lls, beam_toks = lm_probs.topk(beam)
            beam_lls = beam_lls.view(-1, 1)
            beam_toks = beam_toks.view(-1, 1)
            beam_probs = beam_probs.repeat(1, beam).view(-1, 1)
            XMB = XMB.repeat(1, beam, 1).view(-1, XMB.size(1), XMB.size(2))
            mask = mask.repeat(1, beam).view(-1, mask.size(1))
            if _ >= 2:
                seen_hashes = seen_hashes.repeat(1, beam).view(-1, seen_hashes.size(1))
            next_x = torch.cat((beam_toks, XMB[:, -1:, 1] + 1), -1).unsqueeze(1)
            XMB = torch.cat((XMB, next_x), 1)
            beam_probs += beam_lls
            finished_mask = beam_toks.eq(classify_idx)
            #TODO Might be able to batch this
            for i in range(finished_mask.size(0)):
                if finished_mask[i].item() == 1:
                    tokens = []
                    for tok in XMB[i, 1+512+1:-1, 0]:
                        if tok.item() in text_encoder.decoder:
                            tokens.append(text_encoder.decoder[tok.item()].replace('</w>', ' ').replace('\n', ''))
                        else:
                            tokens.append(" <unk> ")
                    phrase = ' '.join(''.join(tokens).split())
                    if min_len is None or len(phrase.split(" ")) >= min_len:
                        finished_beams[i // (beam * beam)].append((XMB[i, 1+512+1:, 0], beam_probs[i].item() / XMB.size(1) - 513)) # 513 to include classify tok in error and avoid division by 0
                    beam_probs[i] = -1e8
        finished_mask = beam_toks.eq(classify_idx)
        beam_seqs = [sorted(finished_beam, key=lambda x: x[1], reverse=True) for finished_beam in finished_beams]
        tokens = torch.zeros(len(beam_seqs), gen_len)
        for i in range(len(beam_seqs)):
            beam_seq = beam_seqs[i][0][0] if len(beam_seqs[i]) != 0 else torch.tensor([classify_idx]).unsqueeze(1).type_as(XMB)
            tokens[i, :beam_seq.size(0)] = beam_seq
        return tokens

    def generate(self, pad_output, mask, text_encoder, device, beam=0, gen_len=110, k=0, decoding_strategy=0, min_len=None):
        classify_idx = text_encoder.encoder['_classify_']
        input_toks = pad_output[:, :1+512+1, 0] # includes delimiter
        target_toks = pad_output[:, -(gen_len+1):, 0]
        mask_pad = torch.ones(mask.size()).type_as(mask)
        mask_pad[:, :1 + 512 + 1] = mask[:, :1 + 512 + 1]
        mask = mask_pad

        pad_output = pad_output.to(device)
        XMB = pad_output[:, :1+512+1]
        if beam == 0:
            generated_toks = self.sample(XMB, mask, classify_idx, text_encoder, gen_len, k, decoding_strategy, min_len=min_len)
        else:
            generated_toks = self.beam_search(XMB, mask, classify_idx, text_encoder, beam=beam, gen_len=gen_len, min_len=min_len)
        return generated_toks.type_as(XMB), input_toks.type_as(XMB), target_toks.type_as(XMB)


def load_openai_pretrained_model(model, n_ctx=-1, n_special=-1, n_transfer=12, n_embd=768, path='./model/',
                                 path_names='./'):
    # Load weights from TF model
    print("Loading weights...")
    names = json.load(open(path_names + 'parameters_names.json'))
    shapes = json.load(open(path + 'params_shapes.json'))
    offsets = np.cumsum([np.prod(shape) for shape in shapes])
    init_params = [np.load(path + 'params_{}.npy'.format(n)) for n in range(10)]
    init_params = np.split(np.concatenate(init_params, 0), offsets)[:-1]
    init_params = [param.reshape(shape) for param, shape in zip(init_params, shapes)]
    if n_ctx > 0:
        init_params[0] = init_params[0][:n_ctx]
    if n_special > 0:
        init_params[0] = np.concatenate(
            [init_params[1],
             (np.random.randn(n_special, n_embd) * 0.02).astype(np.float32),
             init_params[0]
             ], 0)
    else:
        init_params[0] = np.concatenate(
            [init_params[1],
             init_params[0]
             ], 0)
    del init_params[1]
    if n_transfer == -1:
        n_transfer = 0
    else:
        n_transfer = 1 + n_transfer * 12
    init_params = [arr.squeeze() for arr in init_params]

    try:
        assert model.embed.weight.shape == init_params[0].shape
    except AssertionError as e:
        e.args += (model.embed.weight.shape, init_params[0].shape)
        raise

    model.embed.weight.data = torch.from_numpy(init_params[0])

    for name, ip in zip(names[1:n_transfer], init_params[1:n_transfer]):
        name = name[6:]  # skip "model/"
        assert name[-2:] == ":0"
        name = name[:-2]
        name = name.split('/')
        pointer = model
        for m_name in name:
            if re.fullmatch(r'[A-Za-z]+\d+', m_name):
                l = re.split(r'(\d+)', m_name)
            else:
                l = [m_name]
            pointer = getattr(pointer, l[0])
            if len(l) >= 2:
                num = int(l[1])
                pointer = pointer[num]
        try:
            assert pointer.shape == ip.shape
        except AssertionError as e:
            e.args += (pointer.shape, ip.shape)
            raise
        pointer.data = torch.from_numpy(ip)


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


DEFAULT_CONFIG = dotdict({
    'n_embd': 768,
    'n_head': 12,
    'n_layer': 12,
    'embd_pdrop': 0.1,
    'attn_pdrop': 0.1,
    'resid_pdrop': 0.1,
    'afn': 'gelu',
    'clf_pdrop': 0.1})
