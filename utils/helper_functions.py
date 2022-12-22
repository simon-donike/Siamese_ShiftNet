import numpy as np
import matplotlib.pyplot as plt
import torch
import io
import wandb
import PIL


def minmax(img):
    return(img-np.min(img) ) / (np.max(img)-np.min(img))

def minmax_percentile(img,perc=2):
    lower = np.percentile(img,perc)
    upper = np.percentile(img,100-perc)
    img[img>upper] = upper
    img[img<lower] = lower
    return(img-np.min(img) ) / (np.max(img)-np.min(img))

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def plot_tensors(a,b,c,thetas):
    from utils.helper_functions import minmax_percentile
    # A = HR, B = LR, C = Shofted

    "plot with 100x100 window extract"
    a = a.cpu().detach().numpy()[0]
    b = b.cpu().detach().numpy()[0]
    c = c.cpu().detach().numpy()[0]    
    a = np.transpose(a,(1,2,0))
    b = np.transpose(b,(1,2,0))
    c = np.transpose(c,(1,2,0))
    a,b,c = minmax_percentile(a),minmax_percentile(b),minmax_percentile(c)

    
    # calculate theta values for graph
    #thetas_avg_rgb = torch.mean(thetas[:3],0).cpu().detach().numpy()
    #data = {'X': thetas_avg_rgb[0], 'Y': thetas_avg_rgb[1]}
    #names = list(data.keys())
    #values = list(data.values())
    values = thetas.detach().cpu()[0]

    fig, axs = plt.subplots(1, 4,figsize=(20,5),facecolor='white')

    # plot images
    axs[0].imshow(a)
    axs[0].set_title(r"$\bf{"+"HR"+"}$")
    axs[1].imshow(b)
    axs[1].set_title(r"$\bf{"+"SR"+"}$")
    axs[2].imshow(c)
    axs[2].set_title(r"$\bf{"+"Shifted"+"}$")
    
    # draw arrow
    axs[3].arrow(0,0, -1*values[0],-1*values[1],length_includes_head=True,width=0.2)
    axs[3].set_ylim(-10, 10) # set limits at 10 so they stay the same
    axs[3].set_xlim(-10, 10) # set limits at 10 so they stay the same
    axs[3].set_title("Pred. Shifts (px)\nX:  "+str(round(float(values[0]),2))+"\nY:  " + str(round(float(-1*values[1]),2)))
    axs[3].set_xticks([-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9])
    axs[3].set_yticks([-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9])
    axs[3].grid(alpha=0.4) # draw gridlines 
  
    # return wandb image dtype
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    im = PIL.Image.open(buf)
    image = wandb.Image(im, caption="Image")
            
    #return(image)
    wandb.log({"image":image})
    plt.close()
    return None



# plots extra images aswell
def plot_tensors_extra_info(a,b,c,d,e,f,thetas):
    from utils.helper_functions import minmax_percentile
    # A = HR, B = LR, C = Shifted, D= cropped SR

    "plot with 100x100 window extract"
    a = a.cpu().detach().numpy()[0]
    b = b.cpu().detach().numpy()[0]
    c = c.cpu().detach().numpy()[0]    
    a = np.transpose(a,(1,2,0))
    b = np.transpose(b,(1,2,0))
    c = np.transpose(c,(1,2,0))
    a,b,c = minmax_percentile(a),minmax_percentile(b),minmax_percentile(c)
    
    # extra info
    d = d.cpu().detach().numpy()[0] # cropped SR
    e = e.cpu().detach().numpy()[0] # HR Mask
    f = f.cpu().detach().numpy()[0] # SR shifted Mask
    d = np.transpose(d,(1,2,0))
    e = np.transpose(e,(1,2,0))
    f = np.transpose(f,(1,2,0))
    d,e,f = minmax_percentile(d),minmax_percentile(e),minmax_percentile(f)
    
    # calculate theta values for graph
    #thetas_avg_rgb = torch.mean(thetas[:3],0).cpu().detach().numpy()
    #data = {'X': thetas_avg_rgb[0], 'Y': thetas_avg_rgb[1]}
    #names = list(data.keys())
    #values = list(data.values())
    values = thetas.detach().cpu()[0]

    fig, axs = plt.subplots(2, 4,figsize=(20,10),facecolor='white')

    # plot images
    axs[0,0].imshow(a)
    axs[0,0].set_title(r"$\bf{"+"HR"+"}$")
    axs[0,1].imshow(b)
    axs[0,1].set_title(r"$\bf{"+"SR"+"}$")
    axs[0,2].imshow(c)
    axs[0,2].set_title(r"$\bf{"+"Shifted"+"}$")
    
    # draw arrow
    axs[0,3].arrow(0,0, -1*values[0],-1*values[1],length_includes_head=True,width=0.2)
    axs[0,3].set_ylim(-10, 10) # set limits at 10 so they stay the same
    axs[0,3].set_xlim(-10, 10) # set limits at 10 so they stay the same
    axs[0,3].set_title("Pred. Shifts (px)\nX:  "+str(round(float(values[0]),2))+"\nY:  " + str(round(float(-1*values[1]),2)))
    axs[0,3].set_xticks([-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9])
    axs[0,3].set_yticks([-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9])
    axs[0,3].grid(alpha=0.4) # draw gridlines 
    
    # plot images
    axs[1,0].imshow(d)
    axs[1,0].set_title("GT")
    axs[1,1].imshow(e)
    axs[1,1].set_title("SR Pred. Shifted (Loss)")
    axs[1,2].imshow(f)
    axs[1,2].set_title("HR Ground Truth (Loss)")
  
    # return wandb image dtype
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    im = PIL.Image.open(buf)
    image = wandb.Image(im, caption="Image")
    
    wandb.log({"image":image})
    plt.close()
    return None
            
    #return(image)
