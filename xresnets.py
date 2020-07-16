
from torch import nn
import torch
from torch import tensor
import math

def conv_layer(ni,nf,s = 2,ks =3,act = nn.ReLU(), bn = nn.BatchNorm2d, bn_before_act = False,bias = False,**kwargs):
    '''This function simply packages a Convolution, Activation and 
        potentially a BatchNorm layer in a Sequential module'''
    
    layer  = [nn.Conv2d(ni,nf,kernel_size = ks, stride = s, padding = ks//2,bias = bias)]
    if act: layer.append(act)
    if bn : layer.append(bn(nf, **kwargs))
    if act and bn:
        if bn_before_act: layer[1],layer[2] = layer[2],layer[1]
    return nn.Sequential(*layer)

def noop(x): return x

class Flatten(nn.Module):
    '''Flattens the input for.eg if input has shape (32,64,4,4) the output has the shape (32,1024)'''
    def forward(self,x): return x.view(x.size(0), -1)

def init_cnn(func,m):
    '''Initialises the module m with any initialization function you want'''
    if isinstance(m, (nn.Linear, nn.Conv2d)): func(m.weight)
    else:
        for i in m.children(): init_cnn(func, i)


def fixup_init(a,m=2,num_layers=1,mode = 'fan_in',zero_wt = False):
    '''Initalises the layer a with Fixup initialization'''
    if isinstance(a, nn.Conv2d):
        w = a.weight
        if zero_wt:
            nn.init.constant_(w,0)
        else:
            s = w.shape
            c1 = s[1]if mode == 'fan_in' else s[0]
            c = w[0][0].numel()
            std = math.sqrt(2/(c*c1))*num_layers**(-0.5/(m-1))
            nn.init.normal_(w,0, std)

 

class Resblock1(nn.Module):

    '''This class construct a Downsampling block or an Identity block according to the paper
    Bag of Tricks for Image Classification with Convolutional Neural Networks by He et.al'''
    def __init__(self,ni,nf,stride,activation = nn.ReLU(inplace = True),expansion = 4,
                 init = 'kaiming_normal_',num_layers = 1,bn = nn.BatchNorm2d,conv = conv_layer,**kw):
        super().__init__()
        self.init = init
        bn = bn if init != 'Fixup' else False
        nconvs = 2 if expansion ==1 else 3
        if nconvs > 2:
            nh = nf
            ni,nf = expansion*ni, expansion*nf
        nfs = [ni] + [nf]*(nconvs) if nconvs <=2 else [nh]*(nconvs-1) + [nf]
        layer = [conv(ni,nh,s = 1, ks = 1,bn = bn,**kw)] if nconvs > 2 else []
        layer += [conv(nfs[i],nfs[i+1],s = stride if i == 0 else 1, act = None if i == len(nfs)-2 else activation,
                      ks = 1 if (i != 0 and nconvs > 2) else 3, bn = bn, **kw)
                 for i in range(len(nfs)-1)]
        self.ac = activation
        self.convs = nn.Sequential(*layer)
        self.sc = noop if ni == nf else conv(ni,nf,s = 1,ks = 1,act = None, bn = bn,**kw)
        self.pool = noop if stride == 1 else nn.AvgPool2d(kernel_size= 2, stride = 2)
        if init == 'Fixup':
            for i in range(len(self.convs)):
                fixup_init(self.convs[i][0], m = nconvs, num_layers=num_layers,
                           zero_wt = True if i == len(self.convs)-1 else False)

            for i in range(nconvs*2):
                self.register_parameter(f'bias_{i}', nn.Parameter(torch.zeros(1)))
            self.register_parameter('scale', nn.Parameter(torch.ones(1)))


    def forward(self,x):
        if self.init != 'Fixup': return self.ac(self.convs(x) + self.sc(self.pool(x)))
        sc = self.sc(self.pool(x + self.bias_0))
        for i in range(0,len(self.convs)*2, 2):
            k = i//2
            x = self.convs[k][0](x + getattr(self, f'bias_{i}'))
            if len(self.convs[k]) > 1:
                x = self.convs[k][1](x + getattr(self, f'bias_{i+1}'))

        out = self.scale*x + getattr(self, f'bias_{i+1}')
        return self.ac(out + sc)


class ResnetModule(nn.Module):
    def __init__(self,expansions,layers,cin = 3, cout =10,init_func = 'kaiming_normal_',bn = nn.BatchNorm2d,
                 conv = conv_layer,**kw):
        super().__init__()
        nfs = [cin, 32,32,64]
        self.ifunc = init_func
        in_block = [conv(nfs[i], nfs[i+1], s = 2 if i == 0 else 1,
                          bn = False if self.ifunc == 'Fixup' else bn, **kw) for i in range(len(nfs)-1)]
        l = [64//expansions,64,128,256,512]
        blocks = [self._make_blocks(l[i],l[i+1],s = 1 if i == 0 else 2,num = n,init=self.ifunc,expansions = expansions,
                                  num_layers = sum(layers),bn = bn,conv = conv,**kw) for i,n in enumerate(layers)]
        self.lin = nn.Linear(l[-1]*expansions,cout)
        self.m = nn.Sequential(*in_block, nn.MaxPool2d(3,stride = 2,padding = 1),
                          *blocks, nn.AdaptiveAvgPool2d(1), Flatten(),self.lin)
        if self.ifunc == 'Fixup':
            nn.init.constant_(self.lin.weight, 0)
            nn.init.constant_(self.lin.bias,1)
            for i in range(len(in_block)):
                self.m.register_parameter(f'rbias_{i}', nn.Parameter(torch.zeros(1)))
            self.m.register_parameter('linear_bias', nn.Parameter(torch.zeros(1)))
        else:
            i = getattr(nn.init, self.ifunc)
            init_cnn(i,self.m)

    @staticmethod
    def _make_blocks(ni,nf,s,num,init,num_layers,expansions = 1,bn = nn.BatchNorm2d,conv = conv_layer,**kwargs):
        l = [ni] + [nf]*(num)
        return nn.Sequential(*[Resblock1(l[i],l[i+1],stride = s if i == 0 else 1,
                                expansion = expansions,num_layers = num_layers,init = init,bn=bn,conv = conv,**kwargs)
                               for i in range(len(l)-1)])

    def forward(self,x):
        if self.ifunc != 'Fixup': return self.m(x)
        i = 0
        while not isinstance(self.m[i], nn.MaxPool2d):
            x = self.m[i][1](self.m[i][0](x) + getattr(self.m, f'rbias_{i}'))
            i+= 1


        for j in range(i, len(self.m)):
            if isinstance(j, nn.Linear): x = self.m[j](x + getattr(self.m, 'linear_bias'))
            else : x = self.m[j](x)

        return x

def xresnet18(bn = nn.BatchNorm2d,**kwargs): return ResnetModule(1,[2,2,2,2],bn = bn,**kwargs)
def xresnet34(bn = nn.BatchNorm2d,**kwargs): return ResnetModule(1,[3,4,6,3],bn = bn,**kwargs)
def xresnet50(bn = nn.BatchNorm2d,**kwargs): return ResnetModule(4,[3,4,6,3],bn = bn,**kwargs)
def xresnet101(bn = nn.BatchNorm2d,**kwargs): return ResnetModule(4,[3,4,23,3],bn = bn,**kwargs)
def xresnet152(bn = nn.BatchNorm2d,**kwargs): return ResnetModule(4,[3,8,36,3],bn = bn,**kwargs)