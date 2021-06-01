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
ImageFile.LOAD_TRUNCATED_IMAGES = True
import wandb

wandb.init()
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
class CocoDataset():
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""
    def __init__(self, transform=None):
        """Set the path for images, captions and vocabulary wrapper.
        
        Args:
            transform: image transformer.
        """
        super(CocoDataset,self).__init__()
        dataDir = '.'
        dataType = 'val2014'
        #loading instances file into coco variable 
        instances_annFile = os.path.join(dataDir, 'cocoapi/annotations/instances_{}.json'.format(dataType))
        print(instances_annFile)
        self.coco = COCO(instances_annFile)
        # get image ids from instances file
        self.ids = list(self.coco.anns.keys())
        # initialize COCO API for caption annotations
        captions_annFile = os.path.join(dataDir, 'cocoapi/annotations/captions_{}.json'.format(dataType))
        #print(captions_annFile)
        self.coco_caps = COCO(captions_annFile)
        self.ids = list(self.coco.anns.keys())
        self.transform = transform

    def __getitem__(self, index):
        """Returns a of data pair ([image1,image2] and [positive caption,negative caption])."""
        coco = self.coco
        ann_id = self.ids[index]
        ann_id2=self.ids[index+1]
        img_id = coco.anns[ann_id]['image_id']
        img_id2=coco.anns[ann_id2]['image_id']
        img = coco.loadImgs(img_id)[0]
        img2= coco.loadImgs(img_id2)[0]
        url = img['coco_url']
        url2= img2['coco_url']
        urllib.request.urlretrieve(url,'abc')
        image = Image.open('abc')
        urllib.request.urlretrieve(url2,'abc')
        image2 = Image.open('abc')
        annIds = self.coco_caps.getAnnIds(imgIds=img['id'])
        anns = self.coco_caps.loadAnns(annIds)
        texts=anns[0].get('caption')
        texts2=anns[0].get('caption')
        if self.transform is not None:
            image = self.transform(image)
            image2=self.transform(image2)
        texts=clip.tokenize(texts)
        texts2=clip.tokenize(texts2)
        img_list=[image,image2]
        txt_list=[texts,texts2]
        return img_list,txt_list

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
    images,captions = zip(*data)
    final_images=[]
    final_captions=[]
    for i in range(0,len(images)):
        final_images.append(images[i][0])
        final_captions.append(captions[i][0])
        final_images.append(images[i][1])
        final_captions.append(captions[i][1])
    images = torch.stack(final_images, 0)
    lengths = [len(cap[0]) for cap in final_captions]
    targets = torch.zeros(len(final_captions), max(lengths)).long()
    for i, cap in enumerate(final_captions):
        end = lengths[i]
        targets[i, :end] = cap[0][:end]
    return images, targets, lengths

def get_loader(transform, batch_size, shuffle):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    # COCO caption dataset
    coco = CocoDataset(transform=transform)
    
    # Data loader for COCO dataset
    # This will return (images, captions, lengths) for each iteration.
    # images: a tensor of shape (batch_size, 3, 224, 224).
    # captions: a tensor of shape (batch_size, padded_length).
    # lengths: a list indicating valid length for each caption. length is (batch_size).
    data_loader = torch.utils.data.DataLoader(dataset=coco, 
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              collate_fn=collate_fn)
    return data_loader




#############################################################################################################################################
############################################################transform function###############################################################

crop_size=[224,224]
transform = transforms.Compose([ 
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(),
        ])


############################################################################################################################################
###########################################################model weight type converter######################################################
############################################################################################################################################

def convert_models_to_fp32(model): 
    for p in model.parameters(): 
        p.data = p.data.float() 
        p.grad.data = p.grad.data.float() 

############################################################################################################################################
#############################################################Initializing values,model,loss functions and optimizer#########################

batch_size=5
data_loader = get_loader(transform,batch_size,shuffle=False)


device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(device)
model,preprocess=clip.load("ViT-B/32",device=device,jit=False)
if device == "cpu":
  model.float()
else :
  clip.model.convert_weights(model)

loss_img=nn.CrossEntropyLoss()
loss_txt=nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=5e-5,betas=(0.9,0.98),eps=1e-6,weight_decay=0.2)


############################################################################################################################################
##############################################################training code###############################################################
for batch in data_loader:
    images2,texts2,le=batch
    print("training opt")
    optimizer.zero_grad()
    images2=images2.cuda()
    texts2=texts2.cuda()
    print("img,text proc opt")
    if device == "cpu":
        ground_truth = torch.tensor([1,0,1,0,1,0,1,0,1,0]).long().to(device)
    else:
        ground_truth = torch.tensor([1,0,1,0,1,0,1,0,1,0]).long().to(device)

    logits_per_image, logits_per_text = model(images2, texts2)
    total_loss = (loss_img(logits_per_image,ground_truth) + loss_txt(logits_per_text,ground_truth))/2
    print("model opt")
    print(total_loss)
    ###########################################logging using wandb####################################
    wandb.log({"loss":total_loss})
    total_loss.backward()
    if device == "cpu":
         optimizer.step()
    else : 
        convert_models_to_fp32(model)
        optimizer.step()
        clip.model.convert_weights(model)
    ###################################################trail to delete the gpu cache memory since the code
    ###################################################is stopping after 2 batches due to memory issue
    del images2
    torch.cuda.empty_cache()
    del texts2
    torch.cuda.empty_cache()
    del ground_truth
    torch.cuda.empty_cache()

