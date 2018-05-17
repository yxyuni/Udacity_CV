## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 4)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.conv4 = nn.Conv2d(128, 256, 2)
        self.conv5 = nn.Conv2d(256, 512, 2)
        self.pool = nn.MaxPool2d(2)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.3)
        self.dropout4 = nn.Dropout(0.4)
        self.dropout5 = nn.Dropout(0.5)
        self.dropout6 = nn.Dropout(0.5)
        self.dropout7 = nn.Dropout(0.5)
        self.Dense1 = nn.Linear( 12800, 1000)
        self.Dense2 = nn.Linear(1000, 1000)
        self.Dense3 = nn.Linear(1000, 136)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.elu(self.conv1(x)))  # 1, 2, 3, 4
        x = self.dropout1(x) # 5
        x = self.pool(F.elu(self.conv2(x)))  # 6, 7, 8
        x = self.dropout2(x) # 9
        x = self.pool(F.elu(self.conv3(x))) # 10, 11, 12
        x = self.dropout3(x) # 13
        x = self.pool(F.elu(self.conv4(x))) # 14, 15, 16
        x = self.dropout4(x) # 17
        x = self.pool(F.elu(self.conv5(x))) # 
        x = self.dropout5(x) # 
        x = x.view(-1, 12800) # 18
        x = F.elu(self.Dense1(x)) # 19, 20
        x = self.dropout6(x) # 21
        x = F.relu(self.Dense2(x)) # 22, 23
        x = self.dropout7(x)
        x = self.Dense3(x)
        # a modified x, having gone through all the layers of your model, should be returned
        return x
