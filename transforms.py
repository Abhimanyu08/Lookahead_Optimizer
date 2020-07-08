import random
from PIL import ImageEnhance
import numpy as np
import PIL
import torch

class Transform(): _order,_valid = 0, True

def into_rgb(x): return x.convert('RGB')

class ResizeFixed(Transform):
    _order=10
    def __init__(self,size):
        if isinstance(size,int): size=(size,size)
        self.size = size

    def __call__(self, item): return item.resize(self.size, PIL.Image.BILINEAR)


class PilTransform(Transform): _order, _valid = 11,False

class PIL_FLIP(PilTransform):
    def __init__(self,p): self.p = p
    def __call__(self,x): return x.transpose(random.randint(0,6)) if random.random() < self.p else x

class Enhancer(Transform): _order,_valid = 12, False

class BrEnhance(Enhancer):
    def __init__(self): self.en = ImageEnhance.Brightness
    def __call__(self,x): return self.en(x).enhance(random.uniform(0.5,1.5))

class ShEnhance(Enhancer):
    def __init__(self): self.en = ImageEnhance.Sharpness
    def __call__(self,x): return self.en(x).enhance(random.randint(-1,9))

class ConEnhance(Enhancer):
    def __init__(self): self.en = ImageEnhance.Contrast
    def __call__(self,x): return self.en(x).enhance(random.uniform(1,2))

class ColEnhance(Enhancer):
    def __init__(self): self.en = ImageEnhance.Color
    def __call__(self,x): return self.en(x).enhance(random.randint(1,3))

def np_to_float(x): 
    return torch.from_numpy(np.array(x, dtype=np.float32, copy=False)).permute(2,0,1).contiguous()/255.

np_to_float._order = 20