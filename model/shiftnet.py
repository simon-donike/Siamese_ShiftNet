''' Pytorch implementation of HomographyNet.
    Reference: https://arxiv.org/pdf/1606.03798.pdf and https://github.com/mazenmel/Deep-homography-estimation-Pytorch
    Currently supports translations (2 params)
    The network reads pair of images (tensor x: [B,2*C,W,H])
    and outputs parametric transformations (tensor out: [B,n_params]).'''

#source: https://github.com/ServiceNow/HighRes-net/blob/master/src/lanczos.py


import torch
import torch.nn as nn
from lanczos import lanczos_2d as lanczos

class ShiftNet(nn.Module):
    ''' ShiftNet, a neural network for sub-pixel registration and interpolation with lanczos kernel. '''
    
    def __init__(self,in_channel=1):


        '''
        Args:
            in_channel : int, number of input channels
        '''
        #print("Input Channels:",in_channel)
        
        super(ShiftNet, self).__init__()

        self.layer1 = nn.Sequential(nn.Conv2d(2 * in_channel, 64, 3, padding=1),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU())
        self.layer2 = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(),
                                    nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU())
        self.layer4 = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(),
                                    nn.MaxPool2d(2))
        self.layer5 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU())
        self.layer6 = nn.Sequential(nn.Conv2d(128, 128, 3, padding=1),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU(),
                                    nn.MaxPool2d(2))
        self.layer7 = nn.Sequential(nn.Conv2d(128, 128, 3, padding=1),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU())
        self.layer8 = nn.Sequential(nn.Conv2d(128, 128, 3, padding=1),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU())
        self.drop1 = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(128 * 16 * 16, 1024)
        self.activ1 = nn.ReLU()
        self.fc2 = nn.Linear(1024, 2, bias=False)
        self.fc2.weight.data.zero_() # init the weights with the identity transformation

    def forward(self, x):
        '''
        Registers pairs of images with sub-pixel shifts.
        Args:
            x : tensor (B, 2*C_in, H, W), input pairs of images
        Returns:
            out: tensor (B, 2), translation params
        '''

        x[:, 0] = x[:, 0] - torch.mean(x[:, 0], dim=(1, 2)).view(-1, 1, 1)
        x[:, 1] = x[:, 1] - torch.mean(x[:, 1], dim=(1, 2)).view(-1, 1, 1)

        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)

        out = out.view(-1, 128 * 16 * 16)
        out = self.drop1(out)  # dropout on spatial tensor (C*W*H)

        out = self.fc1(out)
        out = self.activ1(out)
        out = self.fc2(out)
        return out

    def transform(self, theta, I, device="cpu"):
        '''
        Shifts images I by theta with Lanczos interpolation.
        Args:
            theta : tensor (B, 2), translation params
            I : tensor (B, C_in, H, W), input images
        Returns:
            out: tensor (B, C_in, W, H), shifted images
        '''

        self.theta = theta
        new_I = lanczos.lanczos_shift(img=I.transpose(0, 1),
                                      shift=self.theta.flip(-1),  # (dx, dy) from register_batch -> flip
                                      a=3, p=5)[:, None]
        return new_I
    
    
def get_thetas(hr,sr,regis_model,n_channels=3):
    #inouts: hr(GT), sr, regis_model
    # crop image
    target_size = 128 # target w & h of image
    middle = sr.shape[2] //2 # get middle of tensor
    offset = target_size //2 # calculate offset needed from middle of tensor
    #n_channels = hr.shape[1]
    #print("n_channels",n_channels)
    hr_small = torch.clone(hr)[:,0:1,middle-offset:middle+offset,middle-offset: middle+offset] # perform crop and keep only 1 band
    sr_small = torch.clone(sr)[:,0:1,middle-offset:middle+offset,middle-offset: middle+offset] # perform crop and keep only 1 band
    #print(f'After Cropping: HR shape: {hr_small.shape}, SR shape: {sr_small.shape}')


    # rearrange from (B,C,W,H) to (B*3,1,W,H)
    hr_small = hr_small.view(-1, 1, 128, 128)
    sr_small = sr_small.view(-1, 1, 128, 128)
    if hr_small.shape!=sr_small.shape:
        print("shape mismatch")
    #print(f'After Cropping & rearranging: HR shape: {hr_small.shape}, SR shape: {sr_small.shape}')


    ## register_batch via network code
    n_views = hr_small.size(1) # get number of views -> amount of images in original, here its 1
    thetas = []
    for i in range(n_views): # iterate over channels (should be 1 in out case)
        theta = regis_model(torch.cat([hr_small[:, i : i + 1], sr_small[:, i : i + 1]], 1)) # send relevant channel to model
        thetas.append(theta)
    thetas = torch.stack(thetas, 1) # stack return
    #print(f'Thetas shape: {thetas.shape}')
    thetas = thetas[:, None, :, :].repeat(1, n_channels, 1, 1) # expand back to 3x channels
    #print(f'Thetas shape after expanding: {thetas.shape}')
    return thetas,hr_small,sr_small


def apply_shifts(sr,thetas,regis_model,n_channels=1):
    # perform translation
    # clone tensors (?)
    shifts = torch.clone(thetas)
    images = torch.clone(sr)
    
    # change names for clarity
    #shifts=thetas
    #images=sr

    ## apply_shift code
    batch_size, n_views, height, width = images.shape
    images = images.view(-1, 1, height, width)
    thetas = thetas.view(-1, 2)

    #print(f'Apply_shift to input shape: {images.shape}, thetas shape: {thetas.shape}')
    # perform translation via built-in function
    new_images = regis_model.transform(thetas, images) # error here
    #print(f'New Images shifted shape: {new_images.shape}')
    # rearrange from (B*C,1,H,W) to (B,3,H,W)
    new_images = new_images.view(-1, n_channels, images.size(2), images.size(3))
    #hr = hr.view(-1, n_channels, images.size(2), images.size(3))
    #print(f'HR: {hr.shape} - ShiftNet ouput: {new_images.shape}')

    return(new_images,thetas)

def get_shift_loss(new_images,hr,loss_func,sr_small,hr_small,target_size=128,relative_loss=False):
    #hr = hr.view(-1, n_channels, images.size(2), images.size(3))
        
    middle = hr.shape[2] //2 # get middle of tensor
    offset = target_size //2 # calculate offset needed from middle of tensor
    try:
        n_channels = sr_small.shape[1]
    except NameError:
        n_channels = 3
        
    new_images_loss = torch.clone(new_images)[:,:,middle-offset:middle+offset,middle-offset: middle+offset]
    hr_loss = torch.clone(hr)[:,:,middle-offset:middle+offset,middle-offset: middle+offset]
    
    loss_before_shift = loss_func(sr_small,hr_small)
    loss_after_shift = loss_func(new_images_loss,hr_loss)
    if relative_loss:
        loss_relative = (1/loss_before_shift)*loss_after_shift
        train_loss = loss_relative
    if not relative_loss:
        train_loss = loss_after_shift
        
    return(train_loss,hr_loss,new_images_loss)