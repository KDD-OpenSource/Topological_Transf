import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
import random
import torch
from matplotlib import image
import matplotlib.image as mpimg
import os
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from PIL import Image

def split_trainings_data_into_train_and_val(train_set,batchsize):
    """
    splits the trainingsset into train, validation and test set using imported parameters test_set_size and val_set_size
    :returns training set dataloder, validation set dataloader, test set dataloader and length of training, validation and test set
    """
    # percentage of training set to use as validation
    num_training= len(train_set)
    indices = list(range(num_training))
    np.random.shuffle(indices)
    split = int(np.floor(0.2 *num_training)) # 0.2 is testsetsize ratio
    train_idx, test_idx = indices[split:], indices[:split]
    # split_val = int((num_training - split)* 0.1) #0.1 is valsetsize ratio
    # train_idx, valid_idx =train_idx[split_val:], train_idx[:split_val]
    # define samplers for obtaining training and validation batches
    train_sampler = SubsetRandomSampler(train_idx)
    #valid_sampler = SubsetRandomSampler(valid_idx)
    test_sampler = SubsetRandomSampler(test_idx)
    train_loader = DataLoader(train_set, batch_size=batchsize,sampler=train_sampler,shuffle=False)
    #valid_loader = DataLoader(train_set, batch_size=batchsize, sampler=valid_sampler, shuffle=False)
    test_loader = DataLoader(train_set, batch_size=batchsize, sampler=test_sampler, shuffle=False)
    return train_loader, test_loader, len(train_idx),  len(test_idx)


class CatDogDataset(Dataset):
    def __init__(self, root, cat_dir, dog_dir, samplesize):
        """
        initializes our class parameters
        """
        self.root = str(root)
        self.samplesize= samplesize
        #print(root+ cat_dir)
        self.catdir= sorted(Path(root+ cat_dir).glob('*.jpeg'))
        self.catdir+= sorted(Path(root+ cat_dir).glob('*.jpg'))
        random.shuffle(self.catdir)
        self.wavpaths= self.catdir[:samplesize // 2]
        self.dogdir= sorted(Path(root + dog_dir).glob('*.jpeg'))
        self.dogdir += sorted(Path(root + dog_dir).glob('*.jpg'))
        random.shuffle(self.dogdir)
        self.wavpaths+= self.dogdir [:samplesize // 2]


    def __getitem__(self, index):
        """
        get item
        """
        # Read image
        image_path = self.wavpaths[index]
        img = mpimg.imread(image_path)
        # plt.imshow(img)
        # plt.show()
        padded = np.resize(img, (224,224,3)) #TODO 224 or 300? Animals_10 300
        #padded = np.zeros((300,300,3))# zero pad to (300,300,3)
        #padded[:img.shape[0], :img.shape[1]] = img
        #plt.imshow(padded)
        #plt.show()
        if index < (self.samplesize//2) :
            label= torch.tensor(1)
        if index >= (self.samplesize //2) and index <= self.samplesize:
            label = torch.tensor(0)
        return padded, label

    def __len__(self):
        return len(self.wavpaths)

    def len(self):
        return self.__len__()

class Construction_of_two_Datasets(Dataset):
    def __init__(self, rootdir_dataset1, rootdir_dataset2, cat_dir_dataset1, cat_dir_dataset2, dog_dir_dataset1, dog_dir_dataset2, samplesize):
        """
        initializes our class parameters
        """
        self.root = str(rootdir_dataset1)
        self.root2= str(rootdir_dataset2)
        self.samplesize= samplesize
        self.catdir1= sorted(Path(rootdir_dataset1+ cat_dir_dataset1).glob('*.jpeg'))
        random.shuffle(self.catdir1)
        self.wavpaths= self.catdir[:samplesize // 2]
        self.dogdir1= sorted(Path(dog_dir_dataset1).glob('*.jpeg'))
        random.shuffle(self.dogdir)
        self.wavpaths+= self.dogdir [:samplesize // 2]
