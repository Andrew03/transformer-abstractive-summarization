import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from newsroom import jsonl

class CustomDataset(Dataset):
    def __init__(self, data_file, encoder, max_size=None, subset=None):
        with jsonl.open(data_file, gzip=True) as f:
            self.data = f.read()
        if subset is not None:
            self.data = [x for x in self.data if x["density_bin"] == subset]
        random.shuffle(self.data)
        if max_size is not None:
            self.data = self.data[:max_size]
        self.encoder = encoder

    def __getitem__(self, index):
        json_data = self.data[index]
        src_phrase = json_data["text"][:512]
        tgt_phrase = json_data["summary"][:110]
        start = torch.LongTensor([self.encoder['_start_']])
        delim = torch.LongTensor([self.encoder['_delimiter_']])
        end = torch.LongTensor([self.encoder['_classify_']])

        pad_output = torch.zeros(512 + 110 + 3, 2).long()
        mask_output = torch.zeros(512 + 110 + 3).long()

        # Tokens
        pad_output[0, 0] = start
        pad_output[1:len(src_phrase)+1, 0] = torch.LongTensor(src_phrase)
        pad_output[1+512, 0] = delim
        pad_output[1+512+1:1+512+1+len(tgt_phrase), 0] = torch.LongTensor(tgt_phrase)
        pad_output[1+512+1+len(tgt_phrase), 0] = end

        # Positional Embedding
        pad_output[1:len(src_phrase)+1, 1] = torch.LongTensor(np.arange(len(self.encoder), len(self.encoder) + len(src_phrase)))
        pad_output[1+512:1+512+1+len(tgt_phrase), 1] = torch.LongTensor(np.arange(len(self.encoder), len(self.encoder) + len(tgt_phrase) + 1))

        # Mask
        mask_output[:1+len(src_phrase)] = torch.ones(1 + len(src_phrase)).long()
        mask_output[1+512+1:1+512+1+len(tgt_phrase)+1] = torch.ones(len(tgt_phrase) + 1).long()
        return pad_output, mask_output

    def __len__(self):
        return len(self.data)

def get_loader(data_file, batch_size, encoder, shuffle=True, num_workers=0, max_size=None, subset=None):
    dataset = CustomDataset(data_file, encoder, max_size=max_size, subset=subset)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=True)
