import argparse
import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefautsHelpFormatter)

parser.add_argument('gtTrainCSV', type=str, default='EmoWild/GT/EmoWild_Valence_Train.csv', help='path to EmoWild train csv')
parser.add_argument('dataDir', type=str, default='EmoWild/Images', help='path to EmoWild images')
parser.add_argument('ofaEncodeDir', type=str, default='EmoWild/OFA_encodings', help='path to EmoWild OFA encodings')
parser.add_argument('bertEncodeDir', type=str, default='EmoWild/BERT_encodings', help='path to EmoWild BERT encodings')
parser.add_argument('batchSize', type=int, default=128, help='Training batch size')
parser.add_argument('lr', type=int, default=0.001, help='Learning rate')
parser.add_argument('gpuIds', type=list, default=[0, 1, 2, 3] help='List of GPU IDs used for training')
parser.add_argument('epochs', type=list, default=20 help='Training epochs')
parser.add_argument('checkpointDir', type=str, default='modelCheckpoints' help='Save checkpoints to checkpointDir')

class EmoWild_Dataset(Dataset):
    def __init__(self, annotations_file, ofa_dir, be_dir):
        self.img = pd.read_csv(annotations_file)
        self.ofa_dir = ofa_dir
        self.be_dir = be_dir

    def __len__(self):
        return len(self.img)

    def __getitem__(self, idx):
        ofa_path = os.path.join(self.ofa_dir, str(self.img.iloc[idx, 1]) + ".jpg" + ".npy")
        be_path = os.path.join(self.be_dir, str(self.img.iloc[idx, 1]) + ".npy")
        ofa_feature = np.load(ofa_path)
        ofa_feature = np.squeeze(ofa_feature)
        be_feature = np.load(be_path)
        label = self.img.iloc[idx, 2] # valence
        return ofa_feature, be_feature, label
    
class FusionNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.aggregate = torch.nn.Conv1d(in_channels=908, out_channels=768, kernel_size=1)
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, ofa_features, be_features):
        ofa_features = self.aggregate(ofa_features)
        x = torch.matmul(be_features, ofa_features)
        x = x[:, -1, :]
        valence = self.linear_relu_stack(x)
        return valence

train_dataloader = EmoWild_Dataset(gtTrainCSV, ofaEncodeDir, bertEncodeDir)
trainloader = torch.utils.data.DataLoader(train_dataloader, batch_size=batchSize, shuffle=True)

# Build the model
fusionNet = FusionNetwork().to(gpuIds[0])
fusionNet = torch.nn.DataParallel(fusionNet, device_ids=gpuIds)
    
# Loss and optimizer
criterion = nn.MSELoss()
params = fusionNet.parameters()
optimizer = torch.optim.Adam(params, lr=lr)


# Train and test at every epoch
# regression

for epoch in range(0, epochs):
    total_train = 0
    count_train = 0
    for feature1, feature2, val in trainloader:
        feature1 = feature1.to(gpuIds[0])
        feature2 = feature2.to(gpuIds[0])
        target = val.to(gpuIds[0])
        
        # forward, backward and optimize
        output = fusionNet(feature1.float(), feature2.float())
        output = output.to(gpuIds[0])
        loss = criterion(output.squeeze(1).float(), target.float())
        total_train = total_train + loss
        count_train += 1
        neuralNet.zero_grad()
        loss.backward()
        optimizer.step()

    print('Epoch [{}/{}], Epoch train Loss: {:.4f}, Epoch Test Loss: {:.4f}'
          .format(epoch+1, '16', total_train/count_train, total/count))

# Save the model checkpoint at the end of training
torch.save(fusionNet.state_dict(), 
           os.path.join(checkpointDir, 'fusionNet_EmoWild_Valence.ckpt'.format(epoch+1)))