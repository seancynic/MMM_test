import math
import random
import numpy as np
import codecs as cs
from tqdm import tqdm
from os.path import join as pjoin
import utils.paramUtil as paramUtil

import torch
from torch.utils.data._utils.collate import default_collate
from torch.utils.data import Dataset, DataLoader


def collate_fn(batch):
    batch.sort(key=lambda x: x[3], reverse=True)
    return default_collate(batch)


'''For use of training text-2-motion generative model'''
class Text2MotionDataset(Dataset):
    def __init__(self, dataset_name, is_test, w_vectorizer, tokenizer_name, codebook_size,
                 max_text_len=20, unit_length=4, shuffle=True, up_low_sep=False):
        
        self.max_length = 20
        self.pointer = 0
        self.dataset_name = dataset_name
        self.up_low_sep = up_low_sep

        self.max_text_len = max_text_len
        self.unit_length = unit_length
        self.w_vectorizer = w_vectorizer

        self.mot_end_idx = codebook_size
        self.mot_pad_idx = codebook_size + 1 # [TODO] I think 513 (codebook_size+1) can be what ever, it will be croped out
        if dataset_name == 't2m':
            self.data_root = './dataset/HumanML3D'
            self.motion_dir = pjoin(self.data_root, 'new_joint_vecs')
            self.text_dir = pjoin(self.data_root, 'texts')
            self.joints_num = 22
            radius = 4
            fps = 20
            self.max_frame_length = 196
            self.max_motion_length = 26 if unit_length == 8 else 50
            dim_pose = 263
            kinematic_chain = paramUtil.t2m_kinematic_chain
            self.meta_dir = 'checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta'
        elif dataset_name == 'kit':
            self.data_root = './dataset/KIT-ML'
            self.motion_dir = pjoin(self.data_root, 'new_joint_vecs')
            self.text_dir = pjoin(self.data_root, 'texts')
            self.joints_num = 21
            radius = 240 * 8
            fps = 12.5
            dim_pose = 251
            self.max_frame_length = 196
            self.max_motion_length = 26 if unit_length == 8 else 50
            kinematic_chain = paramUtil.kit_kinematic_chain
            self.meta_dir = 'checkpoints/kit/VQVAEV3_CB1024_CMT_H1024_NRES3/meta'

        mean = np.load(pjoin(self.meta_dir, 'mean.npy'))
        std = np.load(pjoin(self.meta_dir, 'std.npy'))
        
        if is_test:
            split_file = pjoin(self.data_root, 'test.txt')
        else:
            split_file = pjoin(self.data_root, 'val.txt')

        min_motion_len = 40 if self.dataset_name =='t2m' else 24

        data_dict = {}
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        new_name_list = []
        length_list = []
        for name in tqdm(id_list):
            try:
                motion = np.load(pjoin(self.motion_dir, name + '.npy'))
                if (len(motion)) < min_motion_len or (len(motion) >= 200):
                    continue

                m_token_list = np.load(pjoin(tokenizer_name, f'{name}.npy'))

                text_data = []
                flag = False
                with cs.open(pjoin(self.text_dir, name + '.txt')) as f:
                    for line in f.readlines():
                        text_dict = {}
                        line_split = line.strip().split('#')
                        caption = line_split[0]
                        tokens = line_split[1].split(' ')
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag

                        text_dict['caption'] = caption
                        text_dict['tokens'] = tokens
                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)
                        else:
                            try:
                                n_motion = motion[int(f_tag*fps) : int(to_tag*fps)]
                                if (len(n_motion)) < min_motion_len or (len(n_motion) >= 200):
                                    continue

                                # [INFO] Check with KIT, doesn't come here that mean f_tag & to_tag are 0.0 (tag for caption from-to frames)
                                m_token_list_new = [tokens[int(f_tag * fps / unit_length): int(to_tag * fps / unit_length)] for tokens in
                                                    m_token_list if int(f_tag * fps / unit_length) < int(to_tag * fps / unit_length)]

                                if len(m_token_list_new) == 0:
                                    continue

                                new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                while new_name in data_dict:
                                    new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name

                                data_dict[new_name] = {'motion': n_motion,
                                                       'm_token_list': m_token_list_new,
                                                       'length': len(n_motion),
                                                       'text':[text_dict]}
                                new_name_list.append(new_name)
                                length_list.append(len(n_motion))
                            except:
                                print(line_split)
                                print(line_split[2], line_split[3], f_tag, to_tag, name)
                                # break

                if flag:
                    data_dict[name] = {'motion': motion,
                                       'm_token_list': m_token_list,
                                       'length': len(motion),
                                       'text': text_data}
                    new_name_list.append(name)
                    length_list.append(len(motion))
            except Exception as e:
                # print(e)
                pass

        name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))
        self.mean = mean
        self.std = std
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list
        self.reset_max_len(self.max_length)
        self.shuffle = shuffle

    def reset_max_len(self, length):
        assert length <= self.max_frame_length
        self.pointer = np.searchsorted(self.length_arr, length)
        # print(f'Pointer Pointing at {self.pointer}')
        self.max_length = length

    def inv_transform(self, data):
        return data * self.std + self.mean

    def forward_transform(self, data):
        return (data - self.mean) / self.std

    def __len__(self):
        return len(self.data_dict) - self.pointer

    def __getitem__(self, item):
        idx = self.pointer + item
        name = self.name_list[idx]
        data = self.data_dict[name]
        # data = self.data_dict[self.name_list[idx]]
        m_token_list, motion, m_length, text_list = data['m_token_list'], data['motion'], data['length'], data['text']
        m_tokens = random.choice(m_token_list)
        # Randomly select a caption
        text_data = random.choice(text_list)
        caption, tokens = text_data['caption'], text_data['tokens']

        if len(tokens) < self.max_text_len:
            # pad with "unk"
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
            tokens = tokens + ['unk/OTHER'] * (self.max_text_len + 2 - sent_len)
        else:
            # crop
            tokens = tokens[:self.max_text_len]
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        # get m_tokens, m_tokens_len
        coin = np.random.choice([False, False, True])
        if coin:
            # drop one token at the head or tail
            coin2 = np.random.choice([True, False])
            if coin2:
                m_tokens = m_tokens[:-1]
            else:
                m_tokens = m_tokens[1:]
        m_tokens_len = m_tokens.shape[0]

        if self.up_low_sep:
            new_len = random.randint(20, self.max_motion_length - 1)
            len_mult = math.ceil(new_len / m_tokens_len)
            m_tokens = np.tile(m_tokens, (len_mult, 1))[:new_len]
            m_tokens_len = new_len
            if m_tokens_len + 1 < self.max_motion_length:
                m_tokens = np.concatenate([m_tokens, np.ones((1, 2), dtype=int) * self.mot_end_idx,
                                           np.ones((self.max_motion_length - 1 - m_tokens_len, 2), dtype=int) * self.mot_pad_idx], axis=0)
            else:
                m_tokens = np.concatenate([m_tokens, np.ones((1, 2), dtype=int) * self.mot_end_idx], axis=0)
        else:
            if m_tokens_len + 1 < self.max_motion_length:
                m_tokens = np.concatenate([m_tokens, np.ones((1), dtype=int) * self.mot_end_idx,
                                           np.ones((self.max_motion_length - 1 - m_tokens_len), dtype=int) * self.mot_pad_idx], axis=0)
            else:
                m_tokens = np.concatenate([m_tokens, np.ones((1), dtype=int) * self.mot_end_idx], axis=0)

        # get motion
        if self.unit_length < 10 and self.shuffle:
            coin3 = np.random.choice(['single', 'single', 'double'])
        else:
            coin3 = 'single'

        if coin3 == 'double':
            m_length = (m_length // self.unit_length - 1) * self.unit_length
        elif coin3 == 'single':
            m_length = (m_length // self.unit_length) * self.unit_length
        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx:idx + m_length]

        "Z Normalization"
        motion = (motion - self.mean) / self.std

        if m_length < self.max_frame_length and self.shuffle:
            motion = np.concatenate([motion, np.zeros((self.max_frame_length - m_length, motion.shape[1]))], axis=0)

        return word_embeddings, pos_one_hots, caption, sent_len, m_tokens, motion, m_length, '_'.join(tokens), name

def DATALoader(dataset_name, is_test, batch_size, w_vectorizer, num_workers=8, unit_length=4, shuffle=True):
    val_loader = DataLoader(Text2MotionDataset(dataset_name, is_test, w_vectorizer, unit_length=unit_length, shuffle=shuffle),
                            batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn, drop_last=True)

    return val_loader

def cycle(iterable):
    while True:
        for x in iterable:
            yield x

def collate_fn_with_bert(tokenizer_t, max_t):

    def collate(batch):
        batch.sort(key=lambda x: x[3], reverse=True)
        collated = default_collate(batch)

        word_embeddings, pos_one_hots, captions, sent_len, m_tokens, pose, m_length, _, _ = collated

        with torch.no_grad():
            inputs = tokenizer_t(captions, padding='max_length', truncation=True, max_length=max_t, return_tensors='pt')

        return word_embeddings, pos_one_hots, sent_len, m_tokens, pose, m_length, inputs['input_ids'], inputs['attention_mask'], captions

    return collate

def DATALoaderNew(dataset: str, tokenizer_m: str, w_vectorizer, codebook_size,
                  batch_size=32, is_test=False, tokenizer_t=None, max_t=None, num_w=8, unit_length=4, shuffle=True):
    collate_function = collate_fn_with_bert(tokenizer_t, max_t) if tokenizer_t is not None else collate_fn

    val_loader = DataLoader(Text2MotionDataset(dataset, is_test, w_vectorizer, tokenizer_m, codebook_size, unit_length=unit_length, shuffle=shuffle),
                            batch_size, shuffle=shuffle, num_workers=num_w, collate_fn=collate_function, drop_last=True)

    return val_loader