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

def get_loader(root, json, mode,transform, batch_size, shuffle, num_workers):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    # COCO caption dataset
    coco = CocoDataset(root=root,
                       json=json,
                       transform=transform)
    print(len(coco))
    train_set,test_set,remaning_set=torch.utils.data.random_split(coco,[31900,100,382113])

    # Data loader for COCO dataset
    # This will return (images, captions, lengths) for each iteration.
    # images: a tensor of shape (batch_size, 3, 224, 224).
    # captions: a tensor of shape (batch_size, padded_length).
    # lengths: a list indicating valid length for each caption. length is (batch_size).
    if(mode=="train"):
        data_loader = torch.utils.data.DataLoader(dataset=train_set, 
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)
    elif(mode=="test"):
        data_loader = torch.utils.data.DataLoader(dataset=test_set, 
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)
    return data_loader
############################################################################################################################################
############################################################################################################################################
############################################################################################################################################
#torch.utils.data.random_split(coco,[500,125,20081])


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
    batch_size=10
    num_workers=1
    train="train"
    test="test"
    ####loading a data_loader 
    train_loader = get_loader(image_dir, caption_path, train , 
                             transform, batch_size,
                             shuffle=True, num_workers=num_workers)
    test_loader= get_loader(image_dir, caption_path, test , 
                             transform, batch_size,
                             shuffle=True, num_workers=num_workers)
    print(len(train_loader))
    print(len(test_loader))

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
    optimizer = optim.Adam(model.parameters(), lr=1e-6,betas=(0.9,0.98),eps=1e-7,weight_decay=0.2)
    print(len(train_loader)) 
    epochs=600
    for epoch in range(0,epochs):
        #####batch training
        i=0
        batch_loss=0
        for batch in test_loader:
            i=i+1
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
            print(i,total_loss)
            #loss=total_loss.numpy()
            batch_loss=batch_loss+total_loss.item()
            ###########################################logging using wandb####################################
            #wandb.log({"loss":total_loss},step=i)
            total_loss.backward()
            if device == "cpu":
                optimizer.step()
            else :                                          
                convert_models_to_fp32(model)
                optimizer.step()
                clip.model.convert_weights(model)
        batch_loss=batch_loss/i
        wandb.log({"epoch loss":batch_loss},step=epoch)
        num=0
        for batch in test_loader:
            images,texts=batch
            ##preprocessing images and texts
            images2= torch.stack([transform(img) for img in images],dim=0)
            texts2 = clip.tokenize(texts)
            images2=images2.cuda()
            texts2=texts2.cuda()
            logits_per_image, logits_per_text = model(images2, texts2)
                #probs=logits_per_image.softmax(dim=-1).cuda.numpy()
                #print(probs)
                #print(logits_per_text)

            for j in range(0,len(images)):
                if(max(logits_per_image[j])==logits_per_image[j][j]):
                    num=num+1
        print(num)
        print("accuracy",(num/10000)*100)
        accuracy=num/100
        wandb.log({"epoch testing accuracy":accuracy},step=epoch)
##############################################################################################################################
##############################################################################################################################
##############################################################################################################################

if __name__ == "__main__":
    main()

