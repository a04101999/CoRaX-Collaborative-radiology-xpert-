import torch
from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pdb import set_trace as stop
import os, random
from dataloaders.chexpert_dataset import Chexpert

import warnings
warnings.filterwarnings("ignore")

S
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
    if dataset == 'chexpert':
        coco_root = os.path.join(data_root,'coco')
        ann_dir = data_root
        train_img_root = data_root
        test_img_root = data_root
        train_data_name = 'train.data'
        val_data_name = 'train.data'
        
        train_dataset = Chexpert(
            split='train',
            num_labels=args.num_labels,
            data_file=os.path.join(data_root,train_data_name),
            img_root=train_img_root,
            annotation_dir=ann_dir,
            max_samples=args.max_samples,
            transform=trainTransform,
            known_labels=args.train_known_labels,
            testing=False)
        valid_dataset = Chexpert(split='val',
            num_labels=args.num_labels,
            data_file=os.path.join(data_root,val_data_name),
            img_root=test_img_root,
            annotation_dir=ann_dir,
            max_samples=args.max_samples,
            transform=testTransform,
            known_labels=args.test_known_labels,
            testing=True)

  

    if train_dataset is not None:
        train_loader = DataLoader(train_dataset, batch_size=batch_size,shuffle=True, num_workers=workers,drop_last=drop_last)
    if valid_dataset is not None:
        valid_loader = DataLoader(valid_dataset, batch_size=args.test_batch_size,shuffle=False, num_workers=workers)
    if test_dataset is not None:
        test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size,shuffle=False, num_workers=workers)

    return train_loader,valid_loader,test_loader
