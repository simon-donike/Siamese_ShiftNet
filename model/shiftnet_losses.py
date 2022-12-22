import torch
import kornia

def define_loss(loss_type="MAE"):
    assert loss_type in ['MAE','MSE','SSIM','PSNR']
    if loss_type=="MAE":
        loss_func = torch.nn.functional.l1_loss
    if loss_type=="MSE":
        loss_func = torch.nn.functional.mse_loss
    if loss_type=="SSIM":
        loss_func = kornia.losses.ssim_loss
    if loss_type=="PSNR":
        loss_func = kornia.losses.psnr_loss
        
    #print("Loss func: ",loss_func)
    return(loss_func)