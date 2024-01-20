#!/usr/bin/env python
# coding: utf-8

# In[94]:


get_ipython().system('pip install numpy==1.21')


# In[22]:


cd C-Tran/


# In[23]:



import pandas as pd
args=pd.Series(args)


# In[24]:



import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pickle
from pdb import set_trace as stop
from dataloaders.data_utils import get_unk_mask_indices,image_loader

class Coco80Dataset(Dataset):

    def __init__(self, split,num_labels,data_file,img_root,annotation_dir,max_samples=-1,transform=None,known_labels=0,testing=False,analyze=False):
        self.split=split


        with open(data_file[:-1], "rb") as f:
                 self.split_data=pickle.load(f)
        #self.split_data = pickle.load(open(data_file,'rb'))
        
        if max_samples != -1:
            self.split_data = self.split_data[0:max_samples]

        self.img_root = img_root
        self.transform = transform
        self.num_labels = num_labels
        self.known_labels = known_labels
        self.testing=testing
        self.epoch = 1

    def __len__(self):
        return len(self.split_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_ID = self.split_data[idx]['file_name']

        img_name = os.path.join(self.img_root,image_ID)
        image = image_loader(img_name,self.transform)

        labels = self.split_data[idx]['object']
        labels = torch.Tensor(labels)

        unk_mask_indices = get_unk_mask_indices(image,self.testing,self.num_labels,self.known_labels)
        
        mask = labels.clone()
        mask.scatter_(0,torch.Tensor(unk_mask_indices).long() , -1)

        sample = {}
        sample['image'] = image
        sample['labels'] = labels
        sample['mask'] = mask
        sample['imageIDs'] = image_ID
        return sample



# In[25]:


import torch
from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pdb import set_trace as stop
import os, random

import warnings
warnings.filterwarnings("ignore")


def get_data(args):
    dataset = args.dataset
    data_root=args.dataroot
    batch_size=args.batch_size

    rescale=args.scale_size
    random_crop=args.crop_size
    attr_group_dict=args.attr_group_dict
    workers=args.workers
    n_groups=args.n_groups

    normTransform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    scale_size = rescale
    crop_size = random_crop
    if args.test_batch_size == -1:
        args.test_batch_size = batch_size
    
    trainTransform = transforms.Compose([transforms.Resize((scale_size, scale_size)),
                                        transforms.RandomChoice([
                                        transforms.RandomCrop(640),
                                        transforms.RandomCrop(576),
                                        transforms.RandomCrop(512),
                                        transforms.RandomCrop(384),
                                        transforms.RandomCrop(320)
                                        ]),
                                        transforms.Resize((crop_size, crop_size)),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        normTransform])

    testTransform = transforms.Compose([transforms.Resize((scale_size, scale_size)),
                                        transforms.CenterCrop(crop_size),
                                        transforms.ToTensor(),
                                        normTransform])

    test_dataset = None
    test_loader = None
    drop_last = False
    if dataset == 'coco':
        coco_root = os.path.join(data_root,'coco')
        ann_dir = data_root
        train_img_root = data_root
        test_img_root = data_root
        
        val_data_name = 'val_test.data'
        
        
        valid_dataset = Coco80Dataset(split='val',
            num_labels=args.num_labels,
            data_file=os.path.join(data_root,val_data_name),
            img_root=test_img_root,
            annotation_dir=ann_dir,
            max_samples=args.max_samples,
            transform=testTransform,
            known_labels=args.test_known_labels,
            testing=True)

   

    
    if valid_dataset is not None:
        valid_loader = DataLoader(valid_dataset, batch_size=args.test_batch_size,shuffle=False, num_workers=workers)
   

    return valid_loader

valid_loader=get_data(args)


# In[26]:


import torch
import torch.nn as nn
import argparse,math,numpy as np

from models import CTranModel

from config_args import get_args
import utils.evaluate as evaluate
import utils.logger as logger
from pdb import set_trace as stop
from optim_schedule import WarmupLinearSchedule
from run_epoch import run_epoch



print('Labels: {}'.format(args.num_labels))
print('Train Known: {}'.format(args.train_known_labels))
print('Test Known:  {}'.format(args.test_known_labels))




model = CTranModel(args.num_labels,args.use_lmt,args.pos_emb,args.layers,args.heads,args.dropout,args.no_x_features)
print(model.self_attn_layers)


def load_saved_model(saved_model_name,model):
    checkpoint = torch.load('/home/cougarnet.uh.edu/aawasth3/C-Tran/results/coco.3layer.bsz_16.adam1e-05.lmt.unk_loss/best_model.pt')
    model.load_state_dict(checkpoint['state_dict'])
    return model

#print(args.model_name)
if torch.cuda.device_count() > 1:
    print("Using", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)

model = model.cuda()


# In[27]:



model = load_saved_model(args.saved_model_name,model)
   
data_loader =valid_loader
   
    
all_preds,all_targs,all_masks,all_ids,test_loss,test_loss_unk = run_epoch(args,model,data_loader,None,1,'Testing')
    #test_metrics = evaluate.compute_metrics(args,all_preds,all_targs,all_masks,test_loss,test_loss_unk,0,args.test_known_labels)

 


# In[28]:


x=all_preds


# In[29]:


for o in range(len(x)):
          for n in range(14):
                if x[o][n]<0:
                    x[o][n]=0


# In[30]:


for o in range(len(x)):
          for n in range(14):
                if x[o][n]>1:
                    x[o][n]=1


# In[72]:


import pickle
with open('/home/cougarnet.uh.edu/aawasth3/Eye_Gaze_Research_Data_Set/images/jpg/train.dat', "rb") as f:
                 split_data=pickle.load(f)


# In[73]:


import pickle 
# laod a pickle file
with open("/home/cougarnet.uh.edu/aawasth3/VidChapters/YouCook2/2summary_masked_one_disease_youcook2_asr_align_proc.pkl", "rb") as file:
    loaded_dictm = pickle.load(file)

# display the dictionary
print(loaded_dictm)


# In[74]:


for m in split_data:
    item = 1
    print(m['object'])
 
    if m['object'][0]==1:
        loaded_dictm[m['image_id']]['text']=['No Finding.']


# In[75]:


dise=['No Finding.', 'Enlarged Cardiomediastinum.', 'Cardiomegaly.',
               'Lung Lesion.', 'Lung Opacity.', 'Edema.', 'Consolidation.', 'Pneumonia.',
               'Atelectasis.', 'Pneumothorax.', 'Pleural Effusion.', 'Pleural Other.',
               'Fracture.', 'Support Devices.']
for m in range(len(split_data)):
    item = 1
    print(x[m])
    indices = [i for i in range(len(x[m])) if x[m][i] == item]
    print(indices)
    s=[]
    for b in indices:
        s.append(dise[b])
    split_data[m]['object']=s
    


# In[76]:


split_data


# In[77]:



for f in split_data:
    loaded_dictm[f['image_id']]['text']=list(set(f['object']).union(set(loaded_dictm[f['image_id']]['text'])))
    


# In[78]:




for p in loaded_dictm.items():
    print(p)
    for u in range(len(p[1]['text'])):
        
                
                if p[1]['text'][u][-1]=='.':
                      continue
                else:
                      loaded_dictm[p[0]]['text'][u]=p[1]['text'][u]+'.'
    loaded_dictm[p[0]]['text']=''.join(loaded_dictm[p[0]]['text'])


# In[79]:


loaded_dictm


# In[80]:


cd /home/cougarnet.uh.edu/aawasth3/VidChapters


# In[81]:




# In[82]:


for y in loaded_dictm.items():
    loaded_dictm[y[0]]['start']=[loaded_dictm[y[0]]['start']]
    loaded_dictm[y[0]]['end']=[loaded_dictm[y[0]]['end']]
    loaded_dictm[y[0]]['text']=[loaded_dictm[y[0]]['text']]


# In[83]:


for y in loaded_dictm.items():
    try:
        if loaded_dictm[y[0]]['text'][0][-1]!='.':
              print(loaded_dictm[y[0]]['text'][0])
    except:
        print(y[0])


# In[84]:


loaded_dictm['fdec7e65-a00757e3-4872655f-ee15457b-8da7f362']['text']=['No Finding.']


# In[85]:


loaded_dictm['fdec7e65-a00757e3-4872655f-ee15457b-8da7f362']


# In[86]:


import os
import torch as th
from torch.utils.data import Dataset
import json
import pickle
import numpy as np
from util.t5 import create_sentinel_ids, filter_input_ids, random_spans_noise_mask


class DenseVideoCaptioning_Dataset(Dataset):
    def __init__(
        self,
        json_path,
        features_path,
        max_feats=100,
        features_dim=768,
        tokenizer=None,
        subtitles_path=None,
        num_bins=100,
        max_input_tokens=1000,
        max_output_tokens=256,
        noise_density=0.25,
        mean_noise_span_length=5,
    ):
        self.data = json.load(open(json_path, 'r'))
        self.vids = list(self.data.keys())
        self.features = None
        self.features_path = None
        if os.path.isdir(features_path):
            self.features_path = features_path
        else:
            print(features_path)
            self.features = th.load(features_path)
        self.max_feats = max_feats
        self.features_dim = features_dim
        self.tokenizer = tokenizer
        
        
        self.subs = loaded_dictm
        
        self.num_bins = num_bins
        self.max_input_tokens = max_input_tokens
        self.max_output_tokens = max_output_tokens
        self.num_text_tokens = len(tokenizer) - num_bins
        self.noise_density = noise_density
        self.mean_noise_span_length = mean_noise_span_length

    def __len__(self):
        return len(self.data)

    def _get_text(self, text):
        text = text.strip()
        text = text.capitalize()
        #print(text)
        if text[-1] != '.':
                text = text + '.'
        
        return text

    def _get_video(self, video_id):
        if self.features is not None:
            assert video_id in self.features, video_id
            video = self.features[video_id].float()
        else:
            features_path = os.path.join(self.features_path, video_id + '.mp4.npy')
            if not os.path.exists(features_path):
                features_path = os.path.join(self.features_path, video_id + '.npy')
            assert os.path.exists(features_path), features_path
            video = th.from_numpy(np.load(features_path)).float()

        if len(video) > self.max_feats:
            sampled = []
            for j in range(self.max_feats):
                sampled.append(video[(j * len(video)) // self.max_feats])
            video = th.stack(sampled)
            video_len = self.max_feats
        elif len(video) < self.max_feats:
            video_len = len(video)
            video = th.cat(
                [video, th.zeros(self.max_feats - video_len, self.features_dim)], 0
            )
        else:
            video_len = self.max_feats

        return video

    def time_tokenize(self, x, duration, num_bins):

        time_token = int(float((num_bins - 1) * x) / float(duration))
        
        #print(time_token,self.num_bins)
        if time_token >= self.num_bins:
             print("error")
             print(self.num_bins,x,duration)
             assert time_token <= self.num_bins,duration
        return time_token + self.num_text_tokens

    def __getitem__(self, idx):
        video_id = self.vids[idx]
        annotations = self.data[video_id]
        video = self._get_video(video_id)
        duration = annotations["duration"]

        # get subtitles
        
        if (self.subs is not None and video_id in self.subs):
            sub = self.subs[video_id]
        else:
            print(video_id)

        to_keep = [(x >= 0 and y <= duration) for x, y in zip(sub["start"], sub["end"])]
        if not any(to_keep):  # no subtitles
            input_tokens = (th.ones(1) * self.tokenizer.eos_token_id).long()
        else:
            sub["start"] = [x for i, x in enumerate(sub["start"]) if to_keep[i]]
            sub["end"] = [x for i, x in enumerate(sub["end"]) if to_keep[i]]
            sub['text'] = [self._get_text(x) for i, x in enumerate(sub['text']) if to_keep[i]]
            time_input_tokens = [th.LongTensor([self.time_tokenize(st, duration, self.num_bins),
                                                self.time_tokenize(ed, duration, self.num_bins)])
                                 for st, ed in zip(sub['start'], sub['end'])]
            text_input_tokens = [self.tokenizer(x, add_special_tokens=False, max_length=self.max_input_tokens,
                                                padding="do_not_pad", truncation=True, return_tensors="pt",)['input_ids'][0]
                                 for x in sub['text']]
            input_tokens = [th.cat([ti, te], 0) for ti, te in zip(time_input_tokens, text_input_tokens)]
            input_tokens = th.cat(input_tokens, 0)
            input_tokens = input_tokens[:self.max_input_tokens - 1]
            input_tokens = th.cat([input_tokens, th.LongTensor([self.tokenizer.eos_token_id])], 0)
        
        # denoising sequence
        if len(input_tokens) > 1:
            mask_indices = np.asarray(
                [random_spans_noise_mask(len(input_tokens), self.noise_density, self.mean_noise_span_length)])
            labels_mask = ~mask_indices

            input_ids_sentinel = create_sentinel_ids(mask_indices.astype(np.int8), self.tokenizer, self.num_bins)
            labels_sentinel = create_sentinel_ids(labels_mask.astype(np.int8), self.tokenizer, self.num_bins)

            denoising_output_tokens = th.from_numpy(
                filter_input_ids(input_tokens.unsqueeze(0).numpy(), labels_sentinel, self.tokenizer)).squeeze(0)
            denoising_input_tokens = th.from_numpy(
                filter_input_ids(input_tokens.unsqueeze(0).numpy(), input_ids_sentinel, self.tokenizer)).squeeze(0)
        else:
            input_tokens = th.LongTensor([self.tokenizer.eos_token_id])
            denoising_input_tokens = th.LongTensor([0])
            denoising_output_tokens = input_tokens

        # dvc/vcg sequence
        captions = [self._get_text(x) for x in annotations['sentences']]
        time_output_tokens = [th.LongTensor([self.time_tokenize(st, duration, self.num_bins),
                                             self.time_tokenize(ed, duration, self.num_bins)])
                              for st, ed in annotations['timestamps']]
        text_output_tokens = [self.tokenizer(x, add_special_tokens=False, max_length=self.max_output_tokens,
                                             padding="do_not_pad", truncation=True, return_tensors="pt",)['input_ids'][0]
                              for x in captions]
        output_tokens = [th.cat([ti, te], 0) for ti, te in zip(time_output_tokens, text_output_tokens)]
        
                 
              
        output_tokens = th.cat(output_tokens, 0)
        output_tokens = output_tokens[:self.max_output_tokens - 1]
        output_tokens = th.cat([output_tokens, th.LongTensor([self.tokenizer.eos_token_id])], 0)

        return {
            "video_id": video_id,
            "duration": duration,
            "video": video,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "denoising_input_tokens": denoising_input_tokens,
            "denoising_output_tokens": denoising_output_tokens,
        }


def densevideocaptioning_collate_fn(batch):
    bs = len(batch)
    video_id = [batch[i]["video_id"] for i in range(bs)]
    duration = [batch[i]["duration"] for i in range(bs)]
    video = th.stack([batch[i]["video"] for i in range(bs)])
    input_tokens = [batch[i]["input_tokens"] for i in range(bs)]
    max_input_len = max(len(x) for x in input_tokens)
    for i in range(bs):
        if len(input_tokens[i]) < max_input_len:
            input_tokens[i] = th.cat([input_tokens[i], th.zeros(max_input_len - len(input_tokens[i])).long()], 0)
    input_tokens = th.stack(input_tokens)
    output_tokens = [batch[i]["output_tokens"] for i in range(bs)]
    max_output_len = max(len(x) for x in output_tokens)
    for i in range(bs):
        if len(output_tokens[i]) < max_output_len:
            output_tokens[i] = th.cat([output_tokens[i], th.zeros(max_output_len - len(output_tokens[i])).long()], 0)
    output_tokens = th.stack(output_tokens)
    denoising_input_tokens = [batch[i]["denoising_input_tokens"] for i in range(bs)]
    max_input_len = max(len(x) for x in denoising_input_tokens)
    for i in range(bs):
        if len(denoising_input_tokens[i]) < max_input_len:
            denoising_input_tokens[i] = th.cat(
                [denoising_input_tokens[i], th.zeros(max_input_len - len(denoising_input_tokens[i])).long()], 0)
    denoising_input_tokens = th.stack(denoising_input_tokens)
    denoising_output_tokens = [batch[i]["denoising_output_tokens"] for i in range(bs)]
    max_denoising_output_len = max(len(x) for x in denoising_output_tokens)
    for i in range(bs):
        if len(denoising_output_tokens[i]) < max_denoising_output_len:
            denoising_output_tokens[i] = th.cat([denoising_output_tokens[i], th.zeros(
                max_denoising_output_len - len(denoising_output_tokens[i])).long()], 0)
    denoising_output_tokens = th.stack(denoising_output_tokens)
    out = {
        "video_id": video_id,
        "duration": duration,
        "video": video,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "denoising_input_tokens": denoising_input_tokens,
        "denoising_output_tokens": denoising_output_tokens,
    }
    return out


def build_densevideocaptioning_dataset(dataset_name, split, args, tokenizer):
    if dataset_name == "youcook":
        if split == "train":
            json_path = args.youcook_train_json_path
        elif split == "val":
            json_path = args.youcook_train_json_path
        else:
            raise NotImplementedError
        features_path = args.youcook_features_path
        subtitles_path = args.youcook_subtitles_path
    elif dataset_name == "vitt":
        if split == "train":
            json_path = args.vitt_train_json_path
        elif split == "val":
            json_path = args.vitt_val_json_path
        elif split == "test":
            json_path = args.vitt_test_json_path
        else:
            raise NotImplementedError
        features_path = args.vitt_features_path
        subtitles_path = args.vitt_subtitles_path
    elif dataset_name == "chapters":
        if split == "train":
            json_path = args.chapters_train_json_path
        elif split == "val":
            json_path = args.chapters_val_json_path
        elif split == "test":
            json_path = args.chapters_test_json_path
        else:
            raise NotImplementedError
        features_path = args.chapters_features_path
        subtitles_path = args.chapters_subtitles_path
    else:
        raise NotImplementedError
    return DenseVideoCaptioning_Dataset(json_path=json_path,
                                        features_path=features_path,
                                        max_feats=args.max_feats,
                                        features_dim=args.features_dim,
                                        tokenizer=tokenizer,
                                        subtitles_path=subtitles_path,
                                        num_bins=args.num_bins,
                                        max_input_tokens=args.max_input_tokens,
                                        max_output_tokens=args.max_output_tokens)


# In[87]:


import os
import torch
import numpy as np
import random
import json
import math
import sys
from typing import Iterable
import argparse
import time
import datetime
import re
from util import dist
from torch.utils.data import DataLoader, DistributedSampler
from collections import namedtuple
from functools import reduce

#from dataset import densevideocaptioning_collate_fn, build_densevideocaptioning_dataset, build_yt_dataset, yt_collate_fn
from model import build_vid2seq_model, _get_tokenizer

from util.misc import adjust_learning_rate
from util.metrics import MetricLogger
from dvc_eval import eval_dvc, eval_soda




@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    data_loader,
    device: torch.device,
    args,
    split="test",
    dataset_name="chapters"
):
    model.eval()
    metric_logger = MetricLogger(delimiter="  ")
    header = f"{split}:"

    res = {}

    for i_batch, batch_dict in enumerate(
        metric_logger.log_every(data_loader, args.print_freq, header)
    ):
        duration = batch_dict["duration"]
        video = batch_dict["video"].to(device)
        if "input_tokens" not in batch_dict and args.use_speech:
            input_tokens = torch.ones((video.shape[0], 1)).long().to(device)
            input_tokenized = {'input_ids': input_tokens,
                               'attention_mask': input_tokens != 0}
        elif "input_tokens" in batch_dict:
            input_tokens = batch_dict["input_tokens"].to(device)
            input_tokenized = {'input_ids': input_tokens,
                               'attention_mask': input_tokens != 0}
        else:
            input_tokenized = {'input_ids': None,
                               'attention_mask': None}

        output = model.generate(video=video,
                                input_tokenized=input_tokenized,
                                use_nucleus_sampling=args.num_beams == 0,
                                num_beams=args.num_beams,
                                max_length=args.max_output_tokens,
                                min_length=1,
                                top_p=args.top_p,
                                repetition_penalty=args.repetition_penalty,
                                length_penalty=args.length_penalty,
                                num_captions=1,
                                temperature=1)
        for i, vid in enumerate(batch_dict["video_id"]):
            sequences = re.split(r'(?<!<)\s+(?!>)', output[i]) # "<time=5> <time=7> Blablabla <time=7> <time=9> Blobloblo <time=2>" -> ['<time=5>', '<time=7>', 'Blablabla', '<time=7>', '<time=9>', 'Blobloblo', '<time=2>']
            indexes = [j for j in range(len(sequences) - 1) if sequences[j][:6] == '<time=' and sequences[j + 1][:6] == '<time=']
            last_processed = -2
            res[vid] = []
            for j, idx in enumerate(indexes):  # iterate on predicted events
                if idx == last_processed + 1:  # avoid processing 3 time tokens in a row as 2 separate events
                    continue
                seq = [sequences[k] for k in range(idx + 2, indexes[j + 1] if j < len(indexes) - 1 else len(sequences)) if sequences[k] != '<time=']
                if seq:
                    text = ' '.join(seq)
                else:  # no text
                    continue
                start_re = re.search(r'\<time\=(\d+)\>', sequences[idx])
                assert start_re, sequences[idx]
                start_token = int(start_re.group(1))
                start = float(start_token) * float(duration[i]) / float(args.num_bins - 1)
                end_re = re.search(r'\<time\=(\d+)\>', sequences[idx + 1])
                assert end_re, sequences[idx + 1]
                end_token = int(end_re.group(1))
                end = float(end_token) * float(duration[i]) / float(args.num_bins - 1)
                if end <= start:  # invalid time
                    continue
                res[vid].append({'sentence': text,
                                 'timestamp': [start,
                                               end]})
                last_processed = idx

    all_res = dist.all_gather(res)
    results = reduce(lambda a, b: a.update(b) or a, all_res, {})
    assert len(results) == len(data_loader.dataset)
    metrics = {}
    if dist.is_main_process():
        if args.save_dir:
            pred_path = os.path.join(args.save_dir, dataset_name + f"_{split}_preds.json",)
            json.dump({'results': results}, open(pred_path, "w",))
        else:
            pred_path = {'results': results}
        if dataset_name == "youcook":
            references = [args.youcook_val_json_path]
        elif dataset_name == "vitt":
            references = [args.vitt_val_json_path if split == "val" else args.vitt_test_json_path]
        elif dataset_name == "chapters":
            references = [args.chapters_val_json_path if split == "val" else args.chapters_test_json_path]
        else:
            raise NotImplementedError
        metrics.update(eval_dvc(pred_path, references, tious=[0.3, 0.5, 0.7, 0.9], max_proposals_per_video=1000, verbose=False, no_lang_eval=False))
        metrics.update(eval_soda(pred_path, references, verbose=False))
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")

    metrics = dist.all_gather(metrics)
    metrics = reduce(lambda a, b: a.update(b) or a, metrics, {})

    return metrics

    
  


def main(args):
    # Init distributed mode
    dist.init_distributed_mode(args)

    if dist.is_main_process():
        if args.save_dir and not (os.path.isdir(args.save_dir)):
            os.makedirs(os.path.join(args.save_dir), exist_ok=True)
        print(args)

    device = torch.device(args.device)

    # Fix seeds
    seed = args.seed + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Build model
    tokenizer = _get_tokenizer(args.model_name, args.num_bins)

    nt = namedtuple(
        typename="data",
        field_names=[
            "dataset_name",
            "dataloader_val",
            "dataloader_train",
            "dataloader_test",
        ],
    )

    tuples = []
    for dset_name in args.combine_datasets:
        dataloader_val = None
        dataloader_test = None
        print(dset_name)
        if dset_name in args.combine_datasets_val:
            dataset_val = build_densevideocaptioning_dataset(dset_name, "val", args, tokenizer)
            sampler_val = (
                DistributedSampler(dataset_val, shuffle=False)
                if args.distributed
                else torch.utils.data.SequentialSampler(dataset_val)
            )
            dataloader_val = DataLoader(
                dataset_val,
                batch_size=args.batch_size_val,
                sampler=sampler_val,
                collate_fn=densevideocaptioning_collate_fn,
                num_workers=args.num_workers,
            )
            
            dataloader_test = dataloader_val

       
        dataloader_train = None

        tuples.append(
            nt(
                dataset_name=dset_name,
                dataloader_test=dataloader_test,
                dataloader_val=dataloader_val,
                dataloader_train=dataloader_train,
            )
        )

    model = build_vid2seq_model(args, tokenizer)
    model.to(device)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if dist.is_main_process():
        print("number of params:", n_parameters)
    # print(model)

    # Set up optimizer
    params_for_optimization = list(p for p in model.parameters() if p.requires_grad)
    optimizer = torch.optim.Adam(
        params_for_optimization,
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
    )

    # Load pretrained checkpoint
    if args.load:
        if dist.is_main_process():
            print("loading from", args.load)
        checkpoint = torch.load(args.load, map_location="cpu")
        model.load_state_dict(checkpoint["model"], strict=False)
        if args.resume and not args.eval:
            optimizer.load_state_dict(checkpoint["optimizer"])
            args.start_epoch = checkpoint["epoch"] + 1

    for i, item in enumerate(tuples):
        print(item.dataloader_test)
        out = evaluate(
            model=model,
            data_loader=item.dataloader_test,
            device=device,
            dataset_name=item.dataset_name,
            args=args,
            split="test",
        )

        return out



        args.save_dir = os.path.join(args.presave_dir, args.save_dir)
#args.model_name = os.path.join(os.environ["TRANSFORMERS_CACHE"], args.model_name)
x=main(args)


# In[92]:



path='/home/cougarnet.uh.edu/aawasth3/VidChapters/YouCook2/train.json'
import json

def js_r(filename: str):
    with open(filename) as f_in:
        return json.load(f_in)


# In[93]:


train=js_r(path)


# In[95]:


loaded_dictm['eb9c34a4-dcffac67-d7c42490-bd8699b6-d00422e0']


# In[98]:


for i in loaded_dictm.items():
    if 'No Finding' in i[1]['text'][0]:
        print(i[1]['text'],i[0])
        

