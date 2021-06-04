import os
from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import gzip
import html
import os
from functools import lru_cache
import ftfy
import regex as re
import torch
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image
import requests
from io import BytesIO
import clip
import torch.nn as nn
import torch.optim as optim
import urllib.request
from numpy import asarray
from PIL import Image, ImageFile
from torchvision import transforms
import torch.utils.data as data
ImageFile.LOAD_TRUNCATED_IMAGES = True
import random
import wandb

wandb.init()
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################

class CocoDataset(data.Dataset):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""
    def __init__(self, root, json, transform=None):
        """Set the path for images, captions and vocabulary wrapper.
        
        Args:
            root: image directory.
            json: coco annotation file path.
            vocab: vocabulary wrapper.
            transform: image transformer.
        """
        self.root = root
        self.coco = COCO(json)
        self.ids = list(self.coco.anns.keys())
        self.transform = transform

    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""
        coco = self.coco
        ann_id = self.ids[index]
        caption = coco.anns[ann_id]['caption']
        img_id = coco.anns[ann_id]['image_id']
        path = coco.loadImgs(img_id)[0]['file_name']
        image = Image.open(os.path.join(self.root, path)).convert('RGB')
        target=caption
        return image,target

    def __len__(self):
        return len(self.ids)


def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption).
    
    We should build custom collate_fn rather than using default collate_fn, 
    because merging caption (including padding) is not supported in default.
    Args:
        data: list of tuple (image, caption). 
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.
    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    images, captions = zip(*data)
    targets=captions
    return images, targets

def get_loader(root, json, transform, batch_size, shuffle, num_workers):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    # COCO caption dataset
    coco = CocoDataset(root=root,
                       json=json,
                       transform=transform)
    
    # Data loader for COCO dataset
    # This will return (images, captions, lengths) for each iteration.
    # images: a tensor of shape (batch_size, 3, 224, 224).
    # captions: a tensor of shape (batch_size, padded_length).
    # lengths: a list indicating valid length for each caption. length is (batch_size).
    data_loader = torch.utils.data.DataLoader(dataset=coco, 
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)
    return data_loader
############################################################################################################################################
###########################################################model weight type converter######################################################
############################################################################################################################################

def convert_models_to_fp32(model): 
    for p in model.parameters(): 
        p.data = p.data.float() 
        p.grad.data = p.grad.data.float() 

############################################################################################################################################
############################################################################################################################################
############################################################################################################################################

def main():
    ###assigning the variables of get_loader with respective values
    image_dir="cocoapi/images/train2014/"
    caption_path="./cocoapi/annotations/captions_train2014.json"
    crop_size=[224,224]
    transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(crop_size),
            transforms.RandomHorizontalFlip(), 
            transforms.ToTensor(),
            ])
    batch_size=100
    num_workers=1
    ####loading a data_loader 
    data_loader = get_loader(image_dir, caption_path, 
                             transform, batch_size,
                             shuffle=True, num_workers=num_workers) 
    ####assigning device
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(device)
    ####assigning model
    model,preprocess=clip.load("ViT-B/32",device=device,jit=False)
    ####assigning model weights
    if device == "cpu":
        model.float()
    else :
        clip.model.convert_weights(model)
    ####defining the image,text losses and optimizer
    loss_img=nn.CrossEntropyLoss()
    loss_txt=nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=5e-5,betas=(0.9,0.98),eps=1e-6,weight_decay=0.2)
    
    epochs=2
    for epoch in range(epochs):
        #####batch training
        for batch in data_loader:
            images,texts=batch
            ##preprocessing images and texts
            images2= torch.stack([transform(img) for img in images],dim=0)
            texts2 = clip.tokenize(texts)
            optimizer.zero_grad()
            images2=images2.cuda()
            texts2=texts2.cuda()
            ###tensorizing the labels
            if device == "cpu":
                ground_truth = torch.arrange(len(images)).long().to(device)
            else:
                ground_truth = torch.arange(len(images)).long().to(device)
            logits_per_image, logits_per_text = model(images2, texts2)
            #print(logits_per_image)
            #print(logits_per_text)
            total_loss = (loss_img(logits_per_image,ground_truth) + loss_txt(logits_per_text,ground_truth))/2
            print(total_loss)
            ###########################################logging using wandb####################################
            wandb.log({"batch loss":total_loss})
            total_loss.backward()
            if device == "cpu":
                optimizer.step()
            else :                                          
                convert_models_to_fp32(model)
                optimizer.step()
                clip.model.convert_weights(model)
        wandb.log({"epoch loss":total_loss})

##############################################################################################################################
##############################################################################################################################
##############################################################################################################################

if __name__ == "__main__":
    main()

