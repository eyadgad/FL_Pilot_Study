"""Published attribution controls on the SAME frozen task CNN.
Saliency: Simonyan et al. (2013), gradient absolute value.
Grad-CAM: Selvaraju et al. (2017), activation-weighted localization.
Independent implementations; NOT official authors' released code.
Spatial L1-normalized comparison to absolute IG target.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F
import numpy as np

def saliency(model,images,targets,device):
    model.eval(); x=images.detach().clone().to(device).requires_grad_(True)
    logits=model(x)
    gg=torch.autograd.grad(logits.gather(1,targets.to(device).reshape(-1,1)).sum(),x)[0]
    imp=gg.detach().abs().sum(dim=1).flatten(1)
    imp=imp/(imp.sum(1,keepdim=True)+1.e-12)
    return imp.cpu().numpy()

def _last_conv(model):
    layers=[m for m in model.modules() if isinstance(m,torch.nn.Conv2d)]
    if not layers:raise ValueError('No convolutional layer for Grad-CAM')
    return layers[-1]

def gradcam(model,images,targets,device):
    model.eval(); acts=[];grads=[]
    layer=_last_conv(model)
    def save_act(module,args,output): acts.append(output)
    def save_grad(module,grad_input,grad_output):grads.append(grad_output[0])
    h1=layer.register_forward_hook(save_act)
    h2=layer.register_full_backward_hook(save_grad)
    try:
        xx=images.detach().clone().to(device).requires_grad_(True)
        model.zero_grad(set_to_none=True)
        logits=model(xx)
        logits.gather(1,targets.to(device).reshape(-1,1)).sum().backward()
        a=acts[-1];g=grads[-1]
        alpha=g.mean((2,3),keepdim=True)
        cam=F.relu((alpha*a).sum(1,keepdim=True))
        cam=F.interpolate(cam,size=xx.shape[-2:],mode='bilinear',align_corners=False)
        x=cam.detach().flatten(1)
        x=x/(x.sum(1,keepdim=True)+1.e-12)
        return x.cpu().numpy()
    finally:
        h1.remove();h2.remove()

def collect_published_controls(model,dataset,count,device,batch_size=16):
    from torch.utils.data import DataLoader
    out={'gradient_saliency':[],'gradcam':[]}
    seen=0
    for x,_ in DataLoader(dataset,batch_size=batch_size,shuffle=False,num_workers=0):
        if seen>=count:break
        take=min(len(x),count-seen);x=x[:take]
        with torch.no_grad():pred=model(x.to(device)).argmax(1)
        out['gradient_saliency'].extend(saliency(model,x,pred,device))
        out['gradcam'].extend(gradcam(model,x,pred,device))
        seen+=take
    if seen!=count:raise ValueError('Insufficient original images for published controls')
    return {k:np.asarray(v,dtype=np.float32) for k,v in out.items()}
