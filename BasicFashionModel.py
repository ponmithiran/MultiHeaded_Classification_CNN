#!/usr/bin/env python
# coding: utf-8

# In[3]:


from __future__ import division
import os
import torch.optim as optim
import matplotlib.pyplot as plt
import torch 
import torch.nn as nn
import torchvision.datasets as dsets
from skimage import transform
from skimage import io
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import random
import math
from PIL import Image
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt1
import time

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Intialize start time
start_time = time.time()


# In[12]:


num_epochs = 1;
batch_size = 50;


# In[13]:


class FashionDataset(Dataset):
    
    def __init__(self, purpose, img_size, root_dir):
        self.image_annon=pd.read_csv(f'FashionDataset/split/{purpose}.txt',
                                     sep="\t")

        self.root_dir=root_dir
        
        self.purpose = purpose
        if purpose == "train" or purpose =="val":
            self.labels= np.loadtxt(f'FashionDataset/split/{purpose}_attr.txt', dtype=np.int64)
        
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        if purpose == 'train':
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.ColorJitter(0.5,0.5,0.5,0.5),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        elif purpose == 'val' or purpose =='test':
            self.transform = transforms.Compose([
                transforms.Resize(img_size),
                transforms.ToTensor(),
            ])
        
        
         # read categories
        self.categories = []
        catefn = open('FashionDataset/split/list_attr_cloth.txt').readlines()
        for i, line in enumerate(catefn):
            self.categories.append(line.strip('\n'))

        self.img_size = img_size
        
        #read BBox
        self.bboxes = np.loadtxt(f'FashionDataset/split/{purpose}_bbox.txt', usecols=(0, 1, 2, 3))
        
        #read landmarks
        self.landmarks = np.loadtxt(f'FashionDataset/split/{purpose}_landmards.txt')
        
        
    def __len__(self):
        return len(self.image_annon)
        
    
    def __getitem__(self, index):
        
        img = Image.open(os.path.join('FashionDataset',
                                      self.image_annon.iloc[index,0])).convert('RGB')
        
        width, height = img.size
        
        #BBox
        bbox_cor = self.bboxes[index]
        x1 = max(0, int(bbox_cor[0]) - 10)
        y1 = max(0, int(bbox_cor[1]) - 10)
        x2 = int(bbox_cor[2]) + 10
        y2 = int(bbox_cor[3]) + 10
        bbox_w = x2 - x1
        bbox_h = y2 - y1
        img = img.crop(box=(x1, y1, x2, y2))
        
        img.thumbnail(self.img_size, Image.ANTIALIAS)
        img = self.transform(img)

        if self.purpose == "train" or self.purpose =="val":
            label = torch.from_numpy(self.labels[index])
        else:
            label=[]
        
        #ate = torch.LongTensor([int(self.categories[index]-1)])
        
        #Landmark
        landmark = []
        # compute the shiftness
        origin_landmark = self.landmarks[index]
        for i, l in enumerate(origin_landmark):
            if i % 2 == 0:  # x
                l_x = max(0, l - x1)
                l_x = float(l_x) / bbox_w * self.img_size[0]
                landmark.append(l_x)
            else:  # y
                l_y = max(0, l - y1)
                l_y = float(l_y) / bbox_h * self.img_size[1]
                landmark.append(l_y)
        landmark = torch.from_numpy(np.array(landmark)).float()
            
        data = {'img': img, 'attr': label, 'landmark': landmark}
        
       
        return data


# In[14]:


TrainingDataset= FashionDataset(purpose='train',img_size=[224,224], root_dir='img')
ValDataset= FashionDataset(purpose='val',img_size=[224,224], root_dir='img')
TestDataset= FashionDataset(purpose='test',img_size=[224,224], root_dir='img')


# In[15]:


train_loader = torch.utils.data.DataLoader(dataset=TrainingDataset,
                                           batch_size=batch_size,
                                           shuffle=True)
Val_loader = torch.utils.data.DataLoader(dataset=ValDataset,
                                           batch_size=batch_size,
                                           shuffle=False)
test_loader = torch.utils.data.DataLoader(dataset=TestDataset,
                                           batch_size=batch_size,
                                           shuffle=False)


# In[16]:


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.cate=[]
        self.resnet = torchvision.models.wide_resnet50_2(pretrained=True)
        self.resnet.fc_backup= self.resnet.fc
        self.resnet.fc = nn.Sequential()
        #self.resnet.fc = nn.Sequential(nn.Linear(2048, 512),
#                                  nn.ReLU(),
#                                  nn.Dropout(0.2),
#                                  nn.Linear(512, 10))
        self.fc1 = nn.Linear(self.resnet.fc_backup.in_features, 100)
#         self.fc2 = nn.Linear(100, 20)
        self.fc3_1 = nn.Linear(100, 7)
        self.fc3_2 = nn.Linear(100, 3)
        self.fc3_3 = nn.Linear(100, 3)
        self.fc3_4 = nn.Linear(100, 4)
        self.fc3_5 = nn.Linear(100, 6)
        self.fc3_6 = nn.Linear(100, 3)
        
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        
#         with torch.no_grad():
        x= self.resnet(x)
        x = F.relu(self.fc1(self.dropout(x)))
        cate1=self.fc3_1(x)
        cate2=self.fc3_2(x)
        cate3=self.fc3_3(x)
        cate4=self.fc3_4(x)
        cate5=self.fc3_5(x)
        cate6=self.fc3_6(x)
      
        return cate1,cate2,cate3,cate4,cate5,cate6


# In[17]:


#instance of the Conv Net
model = CNN().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.02, momentum=0.002, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.75)


# In[18]:


n_total_steps = len(train_loader)
def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
    
model.apply(init_weights)
accuracy_score=pd.DataFrame([], columns = ["Val_Error", "Loss","LearningRate"])
for epoch in range(num_epochs):
    for i, (data) in enumerate(train_loader):
    # origin shape: [4, 3, 32, 32] = 4, 3, 1024
    # input_layer: 3 input channels, 6 output channels, 5 kernel size
        images = data['img'].to(device)
        labels = data['attr'].to(device)
        

        model.train()

        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        
        loss1 = criterion(outputs[0], labels[:,0])
        loss2 = criterion(outputs[1], labels[:,1])
        loss3 = criterion(outputs[2], labels[:,2])
        loss4 = criterion(outputs[3], labels[:,3])
        loss5 = criterion(outputs[4], labels[:,4])
        loss6 = criterion(outputs[5], labels[:,5])

        loss = loss1+loss2+loss3+loss4+loss5+loss6

        loss.backward()
        optimizer.step()
        if (i+1) % 20== 0:
            print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')
    loss_plot=loss.item()
    if epoch % 3==0:
        
        model.eval()
        with torch.no_grad():
            tot_acc=0
            for i in range(6):
                n_correct = 0
                n_samples = 0
                for data in Val_loader:
                    images = data['img'].to(device)
                    labels_all = data['attr'].to(device)
                    labels= labels_all[:,i]

                    outputs = model(images)
                    # max returns (value ,index)
                    _,predicted = torch.max(outputs[i], 1)

                    n_samples += labels.size(0)
                    n_correct += (predicted == labels).sum().item()
                acc = 100.0 * n_correct / (n_samples)
                tot_acc+= acc
        #                     print(f'Accuracy of category {i}: {acc} %')
            avg_acc=tot_acc/6
            to_append = [(100.0-avg_acc),loss_plot,optimizer.param_groups[0]["lr"]]
            a_series = pd.Series(to_append, index = accuracy_score.columns)
            accuracy_score = accuracy_score.append(a_series, ignore_index=True)
            print(f'Average accuracy of model: {avg_acc}%')
    
    scheduler.step()
    print(optimizer.param_groups[0]["lr"])


        

print('Finished Training')

plt1.plot(accuracy_score['Val_Error'])
plt1.savefig('Val_Error.pdf')
plt1.clf()

plt1.plot(accuracy_score['Loss'])
plt1.savefig('loss.pdf')
plt1.clf()

plt1.plot(accuracy_score['LearningRate'])
plt1.savefig('LR.pdf')
plt1.clf()

plt1.plot(accuracy_score['LearningRate']*100)
plt1.plot(accuracy_score['Loss'])
plt1.savefig('LR_loss.pdf')
plt1.clf()

accuracy_score.to_csv('Metrics.csv')

PATH = './cnn.pth'
torch.save(model.state_dict(), PATH)


# In[ ]:


model.eval()
with torch.no_grad():
    

    tot_acc=0
    for i in range(6):
        n_correct = 0
        n_samples = 0
        for data in Val_loader:
            images = data['img'].to(device)
            labels_all = data['attr'].to(device)
            labels= labels_all[:,i]
            
            outputs = model(images)
            # max returns (value ,index)
            _,predicted = torch.max(outputs[i], 1)
            
            n_samples += labels.size(0)
            n_correct += (predicted == labels).sum().item()
        acc = 100.0 * n_correct / (n_samples)
        tot_acc+= acc
        print(f'Accuracy of category {i}: {acc} %')
    avg_acc=tot_acc/6
    print(f'Average accuracy of model: {avg_acc}%')


# In[ ]:


model.eval()
with torch.no_grad():
    f = open('test_pred.txt', 'w')
    for data in test_loader:
        images = data['img'].to(device)
        output = model(images)
        
        for i in range(output[0].shape[0]):
            f.write(f'{output[0][i].argmax(0)} '                        
                    f'{output[1][i].argmax(0)} '                        
                    f'{output[2][i].argmax(0)} '                        
                    f'{output[3][i].argmax(0)} '                        
                    f'{output[4][i].argmax(0)} '                        
                    f'{output[5][i].argmax(0)}\n')
f.close()

model_time= time.time()- start_time

print(f'Model Execute Time: {model_time}s')


# In[ ]:




