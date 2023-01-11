import os
import torch
from torch.autograd import Variable


def make_folder(path, version):
        if not os.path.exists(os.path.join(path, version)):
            os.makedirs(os.path.join(path, version))


def tensor2var(x, grad=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=grad)

def var2tensor(x):
    return x.data.cpu()

def var2numpy(x):
    return x.data.cpu().numpy()

def denorm1(x):
    out = (x + 1) / 2
    return out.clamp_(0, 1)

def denorm(v):
    v_min = v.min(axis=2).reshape((v.shape[0],1,1))
    v_max = v.max(axis=2).reshape((v.shape[0],1,1))
    return (v - v_min) / (v_max-v_min)