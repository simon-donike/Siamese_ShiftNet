# ShiftNet  
NN predicting spatial coregistration shift of remote sensing imagery.  
Predicts and applies X and Y affine transforms based on a 128x128 center crop of the input images. Shifts are calculated only on one single band (the first in the stack) and then applied to all bands equally.  
Adapted from [HighResNet - ShiftNet](https://github.com/ServiceNow/HighRes-net/blob/master/src/DeepNetworks/ShiftNet.py "ShiftNet")  

## Data. 
10k 300x300px SPOT6 images. Download [here](https://drive.google.com/file/d/13q9lK8kcdZYkPvDPh_AsSKlRkTe4ercj/view?usp=share_link "Download"), unpack in "data" folder in repo root. Contaings 'train' and 'val' data, out of which only train is used for now.

![Example](images/example2.png "Example")

