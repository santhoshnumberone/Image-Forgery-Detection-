######################## import the libraries #####################################################################
import torch
import torchvision
from torchvision import transforms,datasets
import os
import numpy as np
import matplotlib.pyplot as plt
from torch.optim import lr_scheduler
import time
import copy
import visdom
import torch.nn.init as init
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import torch.nn.init as init
###################################################################################################################




################# create is visdom instance to display the plot for accuracy and loss during the training #########
vis = visdom.Visdom()
###################################################################################################################




################## Used to visualise the loss during the training ####################################
loss_window = vis.line(
    Y=torch.zeros((1)).cpu(),
    X=torch.zeros((1)).cpu(),
    opts=dict(xlabel='epoch',ylabel='Loss',title='training loss, learning_rate=1e-4,step_size=8',legend=['Loss']))
############################################################################################################




################## Used to visualise the accuracy during the training ######################################
acc_window = vis.line(
    Y=torch.zeros((1)).cpu(),
    X=torch.zeros((1)).cpu(),
    opts=dict(xlabel='epoch',ylabel='Acc',title='training acc, learning_rate=1e-4,step_size=8',legend=['Acc']))
############################################################################################################




################### Used to transform the data before feeding it into the neural network ##################
data_transforms = {
        'train': transforms.Compose([
                transforms.RandomResizedCrop(256),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
        ]),
        'val': transforms.Compose([
                transforms.Resize(320),
                transforms.CenterCrop(256),
                transforms.ToTensor()
        ]),
}
############################################################################################################




#################################### Used get the data from the local directory ####################################
data_dir = 'dataset'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])for x in ['train', 'val']}
# Used to control the mini batch size
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=15,shuffle=True, num_workers=15)for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes
#####################################################################################################################




#################################### Use CUDA if GPU present ######################################################
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Get a batch of training data
inputs, classes = next(iter(dataloaders['train']))
# Make a grid from batch
out = torchvision.utils.make_grid(inputs)
###################################################################################################################




######################### [Convolution] -> [Batch Normalization] -> [ReLU] ##################################################
def Conv_block2d(in_channels,out_channels,*args,**kwargs):
	return nn.Sequential(nn.Conv2d(in_channels,out_channels,*args,**kwargs,bias=False),nn.BatchNorm2d(out_channels,eps=0.001),nn.ReLU(inplace=True))
#############################################################################################################################




######################### Inception and ResNet Combined block ###########################################################################
class IRBlock(nn.Module):
    ######################### initilaise the layers required for the block ##################################################
    # INPUTS (in_channels: Number of channels sent into the class, pool_features: Number of features extracted)
    def __init__(self, in_channels, pool_features):
        super(IRBlock,self).__init__()

        #(62x62xin_channels) * (1x1xpool_features) = (62x62xpool_features)
        self.IRB1_Block1_out = Conv_block2d(in_channels,pool_features,kernel_size=1)

        """
        size = [in_channels,pool_features,pool_features,1,3,0,1]
        for in_c,out_c,k,p in zip(size[:2],size[1:3],size[3:5],size[5:]):
            print("in channel",in_c,"out channel",out_c,"kernel",k,"padding",p)
        
        in channel 33 out channel 20 kernel 1 padding 0
        (62x62xin_channels) * (1x1xpool_features) = (62x62xpool_features)
        in channel 20 out channel 20 kernel 3 padding 1
        (62x62xpool_features) * (3x3xpool_features) = (62x62xpool_features)
        """
        self.IRB1_Block2_size = [in_channels,pool_features,pool_features,1,3,0,1]
        self.IRB1_Block2 = [Conv_block2d(in_c,out_c,kernel_size=k,padding=p) for in_c,out_c,k,p in zip(self.IRB1_Block2_size[:2],self.IRB1_Block2_size[1:3],self.IRB1_Block2_size[3:5],self.IRB1_Block2_size[5:])]
        self.IRB1_Block2_out = nn.Sequential(*self.IRB1_Block2)
        #(62x62xpool_features)
        self.IRB1_Block3_out = RNBlock(in_channels,pool_features)

        #(62x62xpool_features) * (1x1xpool_features) = (62x62xpool_features)
        self.IRB1_Block4_out = Conv_block2d(in_channels,pool_features,kernel_size=1)

        # initialise the weight for the Convolution, Linear and Batch Normalization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                # Get the number of inputs for a prticular layer
                in_v,_ = init._calculate_fan_in_and_fan_out(m.weight)
                # initialise the weights randomly with the squareroot of (2/number of inputs for a particular layer)
                X = torch.randn(m.weight.data.size())*np.sqrt(2/in_v)
                # fix the weights of the each layer with the randomly initialise weights
                m.weight.data.copy_(X)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    # define the forward propogation layers in the order of the design  
    def forward(self,x):

        #(62x62xpool_features)
        IRB1_Block1_output = self.IRB1_Block1_out(x)

        #(62x62xpool_features)
        IRB1_Block2_output = self.IRB1_Block2_out(x)

        #(62x62xpool_features)
        IRB1_Block3_output =self.IRB1_Block3_out(x)

        # Average pooling (62x62xin_channels)
        IRB1_Block4_AvgPool_3x3 = F.avg_pool2d(x,kernel_size=3,stride=1,padding=1)
        
        #(62x62xpool_features)
        IRB1_Block4_output = self.IRB1_Block4_out(IRB1_Block4_AvgPool_3x3)
        
        #(62x62x(pool_features x 4))
        output = [IRB1_Block1_output,IRB1_Block2_output,IRB1_Block3_output,IRB1_Block4_output]

        # return the output of the forward propogation
        return torch.cat(output,1)
#############################################################################################################################




############################## ResNet Block ###########################################################################
class RNBlock(nn.Module):
    ######################### initilaise the layers required for the block ##################################################
    # INPUTS (in_channels: Number of channels sent into the class, out_channels: Number of features extracted)
    def __init__(self, in_channels,out_channels):
        super(RNBlock,self).__init__()
        
        #(62x62xout_channels)
        self.RNBlock1 = Conv_block2d(in_channels,out_channels,kernel_size=1)
        #(62x62xout_channels)
        self.RNBlock2 = Conv_block2d(out_channels,out_channels,kernel_size=3,padding=1)
        #(62x62xout_channels)
        self.RNBlock3_3x3 = nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1,bias=False)
        self.RNBlock3_bn1 = nn.BatchNorm2d(out_channels,eps=0.001)


        # initialise the weight for the Convolution, Linear and Batch Normalization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                # Get the number of inputs for a prticular layer
                in_v,_ = init._calculate_fan_in_and_fan_out(m.weight)
                # initialise the weights randomly with the squareroot of (2/number of inputs for a particular layer)
                X = torch.randn(m.weight.data.size())*np.sqrt(2/in_v)
                # fix the weights of the each layer with the randomly initialise weights
                m.weight.data.copy_(X)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    # define the forward propogation layers in the order of the design  
    def forward(self,x):

        # (62x62xout_channels)
        RNBlock1_output = self.RNBlock1(x)

        #Storig value for skip connection after convolution before ReLU
        original_RNBlock1_out = RNBlock1_output

        # (62x62xout_channels)
        RNBlock2_output = self.RNBlock2(RNBlock1_output)

        # (62x62xout_channels)
        RNBlock3_3x3_out = self.RNBlock3_3x3(RNBlock2_output)
        # (62x62xout_channels)
        RNBlock3_bn1_out = self.RNBlock3_bn1(RNBlock3_3x3_out)

        RNBlock3_bn1_out += original_RNBlock1_out
        #(62x62xout_channels)
        RNBlock3_output = F.relu(RNBlock3_bn1_out,inplace=True)

        return RNBlock3_output
########################################################################################################################




############################## Main Model ##########################################################################################
class MainBlock(nn.Module):
    """docstring for MainBlock"nn.Module"""
    def __init__(self):
        super(MainBlock,self).__init__()

        #Maxpooling used in the block 1
        self.pool_2x2 = nn.MaxPool2d(2,2)

        """
        size = [3,15,33,3,3]
        for in_c,out_c,k in zip(size[:2],size[1:3],size[3:]):
            print("in channel",in_c,"out channel",out_c,"kernel size",k)
        
        [Convolution] -> []
        in channel 3 out channel 15 kernel size 3
        (256x256x3) * (3x3x15) = (254x254x15))*(2x2x1) = (127x127x18)
        in channel 15 out channel 33 kernel size 3
        (127x127x15) * (3x3x33) = (127x127x33))*(2x2x1) = (62x62x33)
        """
        self.block1_size = [3,15,33,3,3]
        self.block1_out = [nn.Sequential(*Conv_block2d(in_c,out_c,kernel_size=k),nn.MaxPool2d(2,2)) for in_c,out_c,k in zip(self.block1_size[:2],self.block1_size[1:3],self.block1_size[3:])]
        self.block1_output = nn.Sequential(*self.block1_out)

        # (63x63x80)
        self.block2_output = IRBlock(33,pool_features=20)
        # (62x62x160)
        self.block3_output = IRBlock(80,pool_features=40)
        # (62x62x240)
        self.block4_output = IRBlock(160,pool_features=60)
        # (62x62x360)
        self.block5_output = IRBlock(240,pool_features=90)
        # Linear layer to map the output to the two classes
        self.fc = nn.Linear(62*62*360,2)

        # initialise the weight for the Convolution, Linear and Batch Normalization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                # Get the number of inputs for a prticular layer
                in_v,_ = init._calculate_fan_in_and_fan_out(m.weight)
                # initialise the weights randomly with the squareroot of (2/number of inputs for a particular layer)
                X = torch.randn(m.weight.data.size())*np.sqrt(2/in_v)
                # fix the weights of the each layer with the randomly initialise weights
                m.weight.data.copy_(X)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    # define the forward propogation layers in the order of the design
    def forward(self,x):
    	
        # (62x62x33)
        x = self.block1_output(x)
        # (62x62x80)
        x = self.block2_output(x)
        # (62x62x120)
        x = self.block3_output(x)
        # (62x62x240)
        x = self.block4_output(x)
        # (62x62x360)
        x = self.block5_output(x)
        
        x = x.view(x.size(0),-1)

        x = self.fc(x)
        
        return x
########################################################################################################################




########### Train ########################################################################################################
def train_model(model, criterion, optimizer, scheduler, num_epochs=20):

    # Keep track of the start time of the training
    since = time.time()

    #Load the weights of the model as the best model weights initially
    best_model_wts = copy.deepcopy(model.state_dict())

    #initialise best accuracy of 0.0
    best_acc = 0.0

    # iterate through the number of epochs
    for epoch in range(num_epochs):

        #Print the epoch being iterated through
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        #iterate through train and Validate set
        for phase in ['train','val']:

            # Control the learning rate of the optimizer used and train 
            if phase == ' trian':
                scheduler.step()
                model.train()

            # Run evaluation only if validate
            else:
                model.eval()
            running_loss = 0.0
            running_corrects = 0

            ################### iterate over the entire training and validate data set##############
            for i, (inputs,labels) in enumerate(dataloaders[phase]):
                #load the inputs and labels to CPU/GPU
                inputs = inputs.to(device)
                labels = labels.to(device)

                # clear the gradients for each iteration before calculating backward loss 
                optimizer.zero_grad()

            #forward
            #track history if only in train
                with torch.set_grad_enabled(phase=='train'):
                    outputs = model(inputs)
                    _,preds = torch.max(outputs,1)
                    loss = criterion(outputs,labels)

                    # if training calculate the loss/cost then optimise it
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                #calculate the loss and accuracy 
                running_loss += loss.item()*inputs.size(0)
                running_corrects += torch.sum(preds==labels.data)
            
            epoch_loss = running_loss/dataset_sizes[phase]
            epoch_acc = running_corrects.double()/dataset_sizes[phase]
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))

            #Plotting the epoch loss and epoch accuraucy 
            if phase == 'train':
                # epoch loss
                vis.line(X=torch.ones((1,1)).cpu()*epoch,Y=torch.Tensor([epoch_loss]).unsqueeze(0).cpu(),win=loss_window,update='append')
                # epoch accuracy
                vis.line(X=torch.ones((1,1)).cpu()*epoch,Y=torch.Tensor([epoch_acc]).unsqueeze(0).cpu(),win=acc_window,update='append')

            #deep copy the model
            if phase == 'train' and epoch_acc>best_acc:
                best_acc=epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        
    time_elapsed = time.time()-since
    print('Training complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model
########################################################################################################################




################################# calling the main function ##################################################################
if __name__ == '__main__':
    
    #initilaise the Model
    net = MainBlock().cuda()
    
    # Define the type of loss
    criterion = nn.CrossEntropyLoss()

    # Initilaise the optimizer to be used passing the required parameters of the model and the learning rate 
    optimizer_ft = optim.Adam(net.parameters(),lr=1e-04,betas=(0.9,0.999),eps=1e-08,weight_decay=0,amsgrad=True)

    # Decay learning rate by a factor of 0.1 every 8 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=8, gamma=0.1)

    # Train the model for the defined number of epochs
    model_ft = train_model(net,criterion,optimizer_ft,exp_lr_scheduler,num_epochs=600)
###############################################################################################################################