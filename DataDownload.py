import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

class Dog_Breed_Dataset(Dataset):
    def __init__(self, root_dir, transform=None , train_size = 0.9 , train = True):
        super(Dog_Breed_Dataset, self).__init__()
        
        self.root_dir = root_dir
        self.transform = transform
        self.train_size = train_size
        self.train = train
        
        self.images_path = []
        self.labels = [] 
        
        for index, classes in enumerate(os.listdir(self.root_dir)):
            class_file_path = os.path.join(self.root_dir, classes) 
            for image_filename in os.listdir(class_file_path): 
                full_path = os.path.join(class_file_path, image_filename)
                self.images_path.append(full_path)
                self.labels.append(index)
        
        self.train_images , self.test_images , self.train_labels , self.test_labels = train_test_split(self.images_path , self.labels , train_size = self.train_size , stratify = self.labels , random_state = 42)
        
        if(self.train):
            self.images_path = self.train_images
            self.labels = self.train_labels
        else:
            self.images_path = self.test_images
            self.labels = self.test_labels
                
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        image = Image.open(self.images_path[index]).convert('RGB')
        label = self.labels[index] 
        
        
        if self.transform:
            image = self.transform(image)
        
        label = torch.tensor(label , dtype = torch.long) 
        return image, label

