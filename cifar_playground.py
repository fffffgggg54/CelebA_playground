
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import time
import math
import timm
import timm.optim

def getDataLoader(dataset):
    return torch.utils.data.DataLoader(dataset,
        batch_size = batch_size,
        shuffle=True,
        num_workers=4,
        persistent_workers = True,
        prefetch_factor=2, 
        pin_memory = True, 
        drop_last=True, 
        generator=torch.Generator().manual_seed(41)
    )
    
import numpy as np
def getAccuracy(preds, targs):
    epsilon = 1e-12
    preds = torch.sigmoid(preds)
    targs_inv = 1 - targs
    batchSize = targs.size(dim=0)
    P = targs * preds
    N = targs_inv * preds
    
    
    TP = P.sum(dim=0) / batchSize
    FN = (targs - P).sum(dim=0) / batchSize
    FP = N.sum(dim=0) / batchSize
    TN = (targs_inv - N).sum(dim=0) / batchSize
    
    Precall = TP / (TP + FN + epsilon)
    Nrecall = TN / (TN + FP + epsilon)
    Pprecision = TP / (TP + FP + epsilon)
    Nprecision = TN / (TN + FN + epsilon)
    
    P4 = (4 * TP * TN) / ((4 * TN * TP) + (TN + TP) * (FP + FN) + epsilon)
    
    return torch.column_stack([TP, FN, FP, TN, Precall, Nrecall, Pprecision, Nprecision, P4])
def average_precision(output, target):
    epsilon = 1e-8

    # sort examples
    indices = output.argsort()[::-1]
    # Computes prec@i
    total_count_ = np.cumsum(np.ones((len(output), 1)))

    target_ = target[indices]
    ind = target_ == 1
    pos_count_ = np.cumsum(ind)
    total = pos_count_[-1]
    pos_count_[np.logical_not(ind)] = 0
    pp = pos_count_ / total_count_
    precision_at_i_ = np.sum(pp)
    precision_at_i = precision_at_i_ / (total + epsilon)

    return precision_at_i

def mAP(targs, preds):
    """Returns the model's average precision for each class
    Return:
        ap (FloatTensor): 1xK tensor, with avg precision for each class k
    """

    if np.size(preds) == 0:
        return 0
    ap = np.zeros((preds.shape[1]))
    # compute average precision for each class
    for k in range(preds.shape[1]):
        # sort scores
        scores = preds[:, k]
        targets = targs[:, k]
        # compute average precision
        ap[k] = average_precision(scores, targets)
    return 100 * ap.mean()

# tracking for performance metrics that can be computed from confusion matrix
class MetricTracker():
    def __init__(self):
        self.running_confusion_matrix = None
        self.epsilon = 1e-12
        self.sampleCount = 0
        
    def get_full_metrics(self):
        with torch.no_grad():
            TP, FN, FP, TN = self.running_confusion_matrix / self.sampleCount
            
            Precall = TP / (TP + FN + self.epsilon)
            Nrecall = TN / (TN + FP + self.epsilon)
            Pprecision = TP / (TP + FP + self.epsilon)
            Nprecision = TN / (TN + FN + self.epsilon)
            
            P4 = (4 * TP * TN) / ((4 * TN * TP) + (TN + TP) * (FP + FN) + self.epsilon)
        
            return torch.column_stack([TP, FN, FP, TN, Precall, Nrecall, Pprecision, Nprecision, P4])
        
    def get_aggregate_metrics(self):
        with torch.no_grad():
            TP, FN, FP, TN = (self.running_confusion_matrix / self.sampleCount).mean(dim=1)
            
            Precall = TP / (TP + FN + self.epsilon)
            Nrecall = TN / (TN + FP + self.epsilon)
            Pprecision = TP / (TP + FP + self.epsilon)
            Nprecision = TN / (TN + FN + self.epsilon)
            
            P4 = (4 * TP * TN) / ((4 * TN * TP) + (TN + TP) * (FP + FN) + self.epsilon)
            return torch.stack([TP, FN, FP, TN, Precall, Nrecall, Pprecision, Nprecision, P4])
    
    def update(self, preds, targs):
        self.sampleCount += targs.size(dim=0)
        
        targs_inv = 1 - targs
        P = targs * preds
        N = targs_inv * preds
        
        
        TP = P.sum(dim=0)
        FN = (targs - P).sum(dim=0)
        FP = N.sum(dim=0)
        TN = (targs_inv - N).sum(dim=0)
        
        output = torch.stack([TP, FN, FP, TN])
        if self.running_confusion_matrix is None:
            self.running_confusion_matrix = output
        
        else:
            self.running_confusion_matrix += output
            
        return self.get_aggregate_metrics()

"""# Baseline Models"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from torch.autograd import Variable

__all__ = ['ResNet', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']

def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, in_planes = 16, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = in_planes
        self.num_features = in_planes*(2**(len(num_blocks)-1))
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        '''
        self.layer1 = self._make_layer(block, in_planes, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, in_planes*2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, in_planes*4, num_blocks[2], stride=2)
        #layers = [self.layer1, self.layer2, self.layer3]
        '''
        layers = []
        for layerID, depth in enumerate(num_blocks):
            currLayer = self._make_layer(block, in_planes*(2**(layerID)), num_blocks[layerID], stride=(1 if layerID == 0 else 2))
            layers.append(currLayer)
        
        self.model = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_planes*(2**(len(num_blocks)-1)), num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.model(x)
        #x = self.layer1(x)
        #x = self.layer2(x)
        #x = self.layer3(x)
        x = self.global_pool(x).squeeze()
        x = self.fc(x)
        return x

def resnet6t(**kwargs):
    return ResNet(BasicBlock, [1, 1], in_planes = 8, **kwargs)

def resnet6(**kwargs):
    return ResNet(BasicBlock, [1, 1], **kwargs)

def resnet8t(**kwargs):
    return ResNet(BasicBlock, [1, 1, 1], in_planes = 8, **kwargs)

def resnet8(**kwargs):
    return ResNet(BasicBlock, [1, 1, 1], **kwargs)

def resnet20(**kwargs):
    return ResNet(BasicBlock, [3, 3, 3], **kwargs)


def resnet32(**kwargs):
    return ResNet(BasicBlock, [5, 5, 5], **kwargs)


def resnet44(**kwargs):
    return ResNet(BasicBlock, [7, 7, 7], **kwargs)


def resnet56(**kwargs):
    return ResNet(BasicBlock, [9, 9, 9], **kwargs)


def resnet110(**kwargs):
    return ResNet(BasicBlock, [18, 18, 18], **kwargs)


def resnet1202(**kwargs):
    return ResNet(BasicBlock, [200, 200, 200])

"""# custom models"""

from timm.models.layers import LayerNorm2d, to_2tuple
from torch import Tensor
import torch.nn.functional as F
from torch import linalg as LA


class Stem(nn.Module):
    """ Size-agnostic implementation of 2D image to patch embedding,
        allowing input size to be adjusted during model forward operation
    """

    def __init__(
            self,
            in_chs=3,
            out_chs=96,
            stride=4,
            norm_layer=LayerNorm2d,
    ):
        super().__init__()
        stride = to_2tuple(stride)
        self.stride = stride
        self.in_chs = in_chs
        self.out_chs = out_chs
        assert stride[0] == 4  # only setup for stride==4
        self.conv = nn.Conv2d(
            in_chs,
            out_chs,
            kernel_size=7,
            stride=stride,
            padding=3,
        )
        self.norm = norm_layer(out_chs)

    def forward(self, x: Tensor):
        B, C, H, W = x.shape
        x = F.pad(x, (0, (self.stride[1] - W % self.stride[1]) % self.stride[1]))
        x = F.pad(x, (0, 0, 0, (self.stride[0] - H % self.stride[0]) % self.stride[0]))
        x = self.conv(x)
        x = self.norm(x)
        return x

class ConvPosEnc(nn.Module):
    def __init__(self, dim: int, k: int = 3, act: bool = False):
        super(ConvPosEnc, self).__init__()

        self.proj = nn.Conv2d(dim, dim, k, 1, k // 2, groups=dim)
        self.act = nn.GELU() if act else nn.Identity()

    def forward(self, x: Tensor):
        feat = self.proj(x)
        x = x + self.act(feat)
        return x

# stripped timm impl

from functools import partial

from itertools import repeat
import collections.abc


# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple


def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma

class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            bias=True,
            drop=0.,
            use_conv=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class Tlp(nn.Module):
    """ three layer mlp
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            bias=True,
            drop=0.,
            use_conv=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_3tuple(bias)
        drop_probs = to_3tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act1 = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        
        self.fc2 = linear_layer(hidden_features, hidden_features, bias=bias[1])
        self.act2 = act_layer()
        self.drop2 = nn.Dropout(drop_probs[1])
        
        self.fc3 = linear_layer(hidden_features, out_features, bias=bias[2])
        self.drop3 = nn.Dropout(drop_probs[2])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.drop1(x)
        
        x = self.fc2(x)
        x = self.act2(x)
        x = self.drop2(x)
        
        x = self.fc3(x)
        x = self.drop3(x)
        return x


class TransformerBlock(nn.Module):

    def __init__(
            self,
            dim,
            num_heads=8,
            mlp_ratio=4.,
            qkv_bias=False,
            drop=0.,
            attn_drop=0.,
            init_values=None,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        #self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.mlp = Tlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x

class ViT(nn.Module):
    def __init__(
        self,
        in_chs = 3,
        dim=256,
        num_classes=10,
        depth = 6,
        drop_path = 0.2,
        drop = 0.1
    ):
        super().__init__()

        self.stem = Stem(in_chs = in_chs, out_chs = dim)
        self.cpe = ConvPosEnc(dim=dim, k=3)

        blocks = []

        for i in range(depth):
            blocks.append(
                nn.Sequential(
                    TransformerBlock(dim, num_heads=8, qkv_bias=True, drop_path=drop_path, drop = drop, attn_drop = drop),

                )
            )

        self.blocks = nn.Sequential(*blocks)
        self.depth = depth
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def forward(self,x):
        x = self.stem(x)

        # B, C, H, W -> B, N, C
        x=self.cpe(x).flatten(2).transpose(1, 2)

        x = self.blocks(x)
        
        x = x.mean(dim=1)
        x = self.norm(x)
        x = self.head(x)
        return x


if __name__ == '__main__':
    torch.set_default_dtype(torch.float32)
    #'''
    train_ds = torchvision.datasets.CIFAR100(
        './data/',
        train=True, 
        download=True,
        transform=None
    )
    test_ds = torchvision.datasets.CIFAR100(
        './data/',
        train=False, 
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
        ])
    )
    '''
    train_ds = torchvision.datasets.CIFAR10(
        './data/',
        train=True, 
        download=True,
        transform=None
    )
    test_ds = torchvision.datasets.CIFAR10(
        './data/',
        train=False, 
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
        ])
    )
    '''

    lr = 1e-3
    num_epochs = 100
    batch_size = 512
    grad_acc_epochs = 1
    num_classes = len(train_ds.classes)
    weight_decay = 2e-2

    device = torch.device("cuda:1")

    datasets = {'train':train_ds,'val':test_ds}
    dataloaders = {x: getDataLoader(datasets[x]) for x in datasets}

    """# important stuff"""

    #model = timm.create_model('resnet_10t', pretrained=False, num_classes = num_classes)
    #model = resnet20(num_classes=num_classes)

    model = ViT(dim=256, num_classes=num_classes, depth=8, drop_path=0.1, drop=0.1)


    '''
    import timm.models.vision_transformer

    model = timm.models.vision_transformer.VisionTransformer(
            img_size = 32, 
            patch_size = 4, 
            num_classes = num_classes, 
            embed_dim=128, 
            depth=12, 
            num_heads=1, 
            global_pool='avg', 
            class_token = False, 
            fc_norm=True)
    '''

    model=model.to(device)

    #criterion = nn.BCEWithLogitsLoss()
    criterion = nn.CrossEntropyLoss()
    #optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    optimizer = timm.optim.Adan(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler=torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        lr, 
        epochs=num_epochs, 
        pct_start=0.1, 
        steps_per_epoch=len(dataloaders['train'])
    )

    cycleTime = time.time()
    epochTime = time.time()
    stepsPerPrintout = 10
    for epoch in range(num_epochs):
        
        datasets['train'].transform = transforms.Compose([
            transforms.RandAugment(magnitude = epoch, num_magnitude_bins = int(num_epochs * 1.4)),
            #transforms.RandAugment(),
            transforms.RandomHorizontalFlip(),
            transforms.TrivialAugmentWide(),
            transforms.ToTensor(),
            #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        for phase in ['train', 'val']:
            samples = 0
            correct = 0
            cm_tracker = MetricTracker()
            if phase == 'train':
                model.train()  # Set model to training mode
                #if (hasTPU == True): xm.master_print("training set")
                print("training set")
            else:
                print("test set")
                model.eval()   # Set model to evaluate mode
            for i,(image,labels) in enumerate(dataloaders[phase]):
                image = image.to(device, non_blocking=True)
                labelsOnehot = torch.zeros([batch_size, num_classes]).scatter_(1, labels.view(batch_size, 1), 1).to(device)
                labels = labels.to(device)
                
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(image)
                    preds = torch.argmax(outputs, dim=1)
                    samples += len(image)
                    correct += sum(preds == labels)
                    loss = criterion(outputs,labels)
                    if phase == 'train':
                        loss.backward()
                        if(i % grad_acc_epochs == 0):
                            
                            nn.utils.clip_grad_norm_(
                                model.parameters(), 
                                max_norm=1.0, 
                                norm_type=2
                            )
                            
                            optimizer.step()
                            optimizer.zero_grad()
                        scheduler.step()
                    multiAccuracy = getAccuracy(outputs,labelsOnehot)
                    ma_2 = cm_tracker.update(outputs.softmax(1),labelsOnehot)
                    accuracy = mAP(
                        labelsOnehot.numpy(force=True),
                        outputs.sigmoid().numpy(force=True)
                    )
                    top1 = 100 * (correct/(samples+1e-8))
                    if i % stepsPerPrintout == 0:
                        imagesPerSecond = (batch_size * stepsPerPrintout)/(time.time() - cycleTime)
                        cycleTime = time.time()
                        print('[%d/%d][%d/%d]\tLoss: %.4f\tImages/Second: %.4f\tmAP: %.2f\ttop-1: %.2f\tP4: %.2f\tP4: %.2f' % (epoch,
                            num_epochs,
                            i,
                            len(dataloaders[phase]),
                            loss,
                            imagesPerSecond,
                            accuracy,
                            top1,
                            multiAccuracy[:,8].mean() * 100,
                            ma_2[8] * 100
                        ))

            #print(cm_tracker.get_full_metrics())
            print(cm_tracker.get_aggregate_metrics())
            print(f'top-1: {100 * (correct/samples)}%')
        print(f'finished epoch {epoch} in {time.time()-epochTime}')
        epochTime = time.time()