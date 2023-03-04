import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import timm
import timm.layers.ml_decoder as ml_decoder
import time
import timm.optim

import numpy as np



from typing import Optional

import torch
from torch import nn
from torch import nn, Tensor
from torch.nn.modules.transformer import _get_activation_fn
import mz

import pandas as pd



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


def stepAtThreshold(x, threshold, k=10, base=1000):
    return 1 / (1 + torch.pow(base, (0 - k) * (x - threshold)))

def zero_grad(p, set_to_none=False):
    if p.grad is not None:
        if set_to_none:
            p.grad = None
        else:
            if p.grad.grad_fn is not None:
                p.grad.detach_()
            else:
                p.grad.requires_grad_(False)
            p.grad.zero_()
    return p

class getDecisionBoundary(nn.Module):
    def __init__(self, initial_threshold = 0.5, lr = 1e-3, threshold_min = 0.2, threshold_max = 0.8):
        super().__init__()
        self.initial_threshold = initial_threshold
        self.thresholdPerClass = None
        self.needs_init = True
        self.lr = lr
        self.threshold_min = threshold_min
        self.threshold_max = threshold_max
        
    def forward(self, preds, targs):
        if self.needs_init:
            classCount = preds.size(dim=1)
            currDevice = preds.device
            if self.thresholdPerClass == None:
                self.thresholdPerClass = torch.ones(classCount, device=currDevice, requires_grad=True).to(torch.float64) * self.initial_threshold
            else:
                self.thresholdPerClass = torch.ones(classCount, device=currDevice, requires_grad=True).to(torch.float64) * self.thresholdPerClass
            self.needs_init = False
        
        # need fp64
        self.thresholdPerClass.retain_grad()
        self.thresholdPerClass = self.thresholdPerClass.to(torch.float64)
        if preds.requires_grad:
            preds = preds.detach()
            
            predsModified = stepAtThreshold(preds, self.thresholdPerClass)
            metrics = getAccuracy(predsModified, targs)

            numToMax = metrics[:,8].sum()

            # TODO clean up this optimization phase
            numToMax.backward()
            with torch.no_grad():
                new_threshold = self.lr * self.thresholdPerClass.grad
                self.thresholdPerClass.add_(new_threshold)
                #self.thresholdPerClass = self.thresholdPerClass.clamp(min=self.threshold_min, max=self.threshold_max)
            
            self.thresholdPerClass = zero_grad(self.thresholdPerClass)
            self.thresholdPerClass = self.thresholdPerClass.detach()
            self.thresholdPerClass.requires_grad=True
        return self.thresholdPerClass.detach()

class Hill(nn.Module):
    r""" Hill as described in the paper "Robust Loss Design for Multi-Label Learning with Missing Labels "
    .. math::
        Loss = y \times (1-p_{m})^\gamma\log(p_{m}) + (1-y) \times -(\lambda-p){p}^2 
    where : math:`\lambda-p` is the weighting term to down-weight the loss for possibly false negatives,
          : math:`m` is a margin parameter, 
          : math:`\gamma` is a commonly used value same as Focal loss.
    .. note::
        Sigmoid will be done in loss. 
    Args:
        lambda (float): Specifies the down-weight term. Default: 1.5. (We did not change the value of lambda in our experiment.)
        margin (float): Margin value. Default: 1 . (Margin value is recommended in [0.5,1.0], and different margins have little effect on the result.)
        gamma (float): Commonly used value same as Focal loss. Default: 2
    """

    def __init__(self, lamb: float = 1.5, margin: float = 1.0, gamma: float = 2.0,  reduction: str = 'sum') -> None:
        super(Hill, self).__init__()
        self.lamb = lamb
        self.margin = margin
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        call function as forward
        Args:
            logits : The predicted logits before sigmoid with shape of :math:`(N, C)`
            targets : Multi-label binarized vector with shape of :math:`(N, C)`
        Returns:
            torch.Tensor: loss
        """

        # Calculating predicted probability
        logits_margin = logits - self.margin
        pred_pos = torch.sigmoid(logits_margin)
        pred_neg = torch.sigmoid(logits)

        # Focal margin for postive loss
        pt = (1 - pred_pos) * targets + (1 - targets)
        focal_weight = pt ** self.gamma

        # Hill loss calculation
        los_pos = targets * torch.log(pred_pos)
        los_neg = (1-targets) * -(self.lamb - pred_neg) * pred_neg ** 2

        loss = -(los_pos + los_neg)
        loss *= focal_weight

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True):
        super(AsymmetricLoss, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w

        return -loss.sum()

class SPLCModified(nn.Module):

    def __init__(
        self,
        tau: float = 0.6,
        change_epoch: int = 1,
        margin: float = 1.0,
        gamma: float = 2.0,
        reduction: str = 'sum',
        loss_fn: nn.Module = AsymmetricLoss(gamma_neg=0, gamma_pos=0, clip=0.0)
    ) -> None:
        super().__init__()
        self.tau = tau
        self.tau_per_class = None
        self.change_epoch = change_epoch
        self.margin = margin
        self.gamma = gamma
        self.reduction = reduction
        self.loss_fn = loss_fn
        if hasattr(self.loss_fn, 'reduction'):
            self.loss_fn.reduction = self.reduction


    def forward(
        self, 
        logits: torch.Tensor, 
        targets: torch.LongTensor,
        epoch
    ) -> torch.Tensor:
        if self.tau_per_class == None:
            classCount = logits.size(dim=1)
            currDevice = logits.device
            self.tau_per_class = torch.ones(classCount, device=currDevice) * self.tau

        # Subtract margin for positive logits
        logits = torch.where(targets == 1, logits-self.margin, logits)
        
        pred = torch.sigmoid(logits)

        # SPLC missing label correction
        if epoch >= self.change_epoch:
            targets = torch.where(
                pred > self.tau_per_class,
                torch.tensor(1).to(pred), 
                targets
            )

        loss = self.loss_fn(logits, targets)

        return loss

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
        return self.get_full_metrics().mean(dim=0)
    
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

lr = 3e-3
lr_warmup_epochs = 5
num_epochs = 100
batch_size = 256
grad_acc_epochs = 1
num_classes = 40
weight_decay = 2e-3
resume_epoch = 0

device = 'cuda:0'

def getDataLoader(dataset):
    return torch.utils.data.DataLoader(dataset,
        batch_size = batch_size,
        shuffle=True,
        num_workers=3,
        persistent_workers = True,
        prefetch_factor=2, 
        pin_memory = True, 
        drop_last=True, 
        generator=torch.Generator().manual_seed(41)
    )


if __name__ == '__main__':



    train_ds = torchvision.datasets.CelebA(
        './data/',
        'train', 
        download=True,
        transform=transforms.Compose([
            transforms.Resize((64,64)),
            transforms.RandAugment(),
            #transforms.TrivialAugmentWide(),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
    )
    test_ds = torchvision.datasets.CelebA(
        './data/',
        'valid', 
        download=True,
        transform=transforms.Compose([
            transforms.Resize((64,64)),
            transforms.ToTensor(),
        ])
    )
    tagNames = pd.read_csv('./data/celeba/list_attr_celeba.txt', header=1, delim_whitespace=True).columns.values.tolist()


    datasets = {'train':train_ds,'val':test_ds}
    dataloaders = {x: getDataLoader(datasets[x]) for x in datasets}

    #model = timm.create_model('mobilenetv3_small_100', pretrained=False, num_classes = num_classes)
    #model = timm.create_model('mobilenetv3_small_050', pretrained=False, num_classes = num_classes)
    '''
    model = timm.models.DaViT(
        in_chans=3,
        depths=(1, 1, 3, 1),
        patch_size=4,
        embed_dims=(96, 192, 384, 768),
        num_heads=(3, 6, 12, 24),
        window_size=7,
        mlp_ratio=4.,
        qkv_bias=True,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        attention_types=('spatial', 'channel'),
        ffn=True,
        overlapped_patch=False,
        cpe_act=False,
        drop_rate=0.,
        attn_drop_rate=0.,
        num_classes=1000,
        global_pool='avg'
    )
    
    model = timm.models.DaViT(
        in_chans=3,
        depths=(1, 1, 3, 1),
        patch_size=4,
        embed_dims=(8, 16, 32, 64),
        num_heads=(1, 2, 4, 8),
        window_size=7,
        mlp_ratio=4.,
        qkv_bias=True,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        attention_types=('spatial', 'channel'),
        ffn=True,
        overlapped_patch=False,
        cpe_act=False,
        drop_rate=0.,
        attn_drop_rate=0.,
        num_classes=num_classes,
        global_pool='avg'
    )
    '''
    
    model = mz.resnet8(num_classes = num_classes)
    
    '''
    model = mz.ViT(
        image_size = 64,
        patch_size = 4,
        num_classes = num_classes,             # number of stages
        dim = 128,  # dimensions at each stage
        depth = 4,              # transformer of depth 4 at each stage
        heads = 4,      # heads at each stage
        mlp_dim = 256,
        dropout = 0.25,
        dim_head = 32
    )
    '''
    #model = mz.add_ml_decoder_head(model)
    
    
    
    if (resume_epoch > 0):
        model.load_state_dict(torch.load('./models/saved_model_epoch_' + str(resume_epoch - 1) + '.pth'))
        for param in model.parameters():
            param.requires_grad = True
    
    model=model.to(device)
    criterion = AsymmetricLoss(gamma_neg=0, gamma_pos=0, clip=0.0)
    #criterion = SPLCModified()
    #criterion = Hill()
    #criterion = nn.BCEWithLogitsLoss()
    #optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    optimizer = timm.optim.Adan(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
        max_lr=lr, 
        steps_per_epoch=len(dataloaders['train']),
        epochs=num_epochs, 
        pct_start=lr_warmup_epochs/num_epochs
    )
    
    boundaryCalculator = getDecisionBoundary()
    
    scheduler.last_epoch = len(dataloaders['train'])*resume_epoch
    cycleTime = time.time()
    epochTime = time.time()
    stepsPerPrintout = 50
    for epoch in range(resume_epoch, num_epochs):
        AP_regular = []
        AccuracyRunning = []
        for phase in ['train', 'val']:
            cm_tracker = MetricTracker()
            cm_tracker_unmod = MetricTracker()
            if phase == 'train':
                model.train()  # Set model to training mode
                #if (hasTPU == True): xm.master_print("training set")
                print("training set")
            else:
                torch.save(model.state_dict(), './models/saved_model_epoch_' + str(epoch) + '.pth')
                model.eval()   # Set model to evaluate mode
                
                print("validation set")
            for i,(image,labels) in enumerate(dataloaders[phase]):
                image = image.to(device, non_blocking=True)
                labels = labels.float().to(device)
                
                with torch.set_grad_enabled(phase == 'train'):

                    outputs = model(image)
                    
                    
                    preds = torch.sigmoid(outputs)
                    boundary = boundaryCalculator(preds, labels)
                    predsModified = (preds > boundary).float()
                    multiAccuracy = cm_tracker.update(predsModified, labels)
                    multiAccuracyUnmod = cm_tracker_unmod.update(preds, labels)
                    accuracy = mAP(
                        labels.numpy(force=True),
                        outputs.sigmoid().numpy(force=True)
                    )
                    #loss = criterion(outputs,labels)
                    criterion.tau_per_class = boundary + 0.1
                    loss = criterion(outputs, labels, epoch)
                    if loss.isnan():
                        print(outputs.cpu())
                        print(outputs.cpu().sigmoid())
                        exit()
                    if phase == 'train':
                        loss.backward()
                        if(i % grad_acc_epochs == 0):
                            '''
                            nn.utils.clip_grad_norm_(
                                model.parameters(), 
                                max_norm=1.0, 
                                norm_type=2
                            )
                            '''
                            optimizer.step()
                            optimizer.zero_grad()
                        scheduler.step()
                    
                    if i % stepsPerPrintout == 0:
                        
                        imagesPerSecond = (batch_size * stepsPerPrintout)/(time.time() - cycleTime)
                        cycleTime = time.time()
                        torch.set_printoptions(linewidth = 200, sci_mode = False)
                        print(f"[{epoch}/{num_epochs}][{i}/{len(dataloaders[phase])}]\tLoss: {loss:.4f}\tImages/Second: {imagesPerSecond:.4f}\tAccuracy: {accuracy:.2f}\t {[f'{num:.2f}' for num in (multiAccuracy * 100).tolist()]}\t{[f'{num:.2f}' for num in (multiAccuracyUnmod * 100).tolist()]}")
                        torch.set_printoptions(profile='default')
                    
                    if phase == 'val':
                        AP_regular.append(accuracy)
                        AccuracyRunning.append(multiAccuracy)
            
            if (phase == 'val'):
                #torch.set_printoptions(profile="full")
                AvgAccuracy = cm_tracker.get_full_metrics()
                AvgAccuracyUnmod = cm_tracker_unmod.get_full_metrics()
                LabelledAccuracy = list(zip(AvgAccuracy.tolist(), AvgAccuracyUnmod.tolist(), tagNames, boundaryCalculator.thresholdPerClass))
                LabelledAccuracySorted = sorted(LabelledAccuracy, key = lambda x: x[0][8], reverse=True)
                MeanStackedAccuracy = cm_tracker.get_aggregate_metrics()
                MeanStackedAccuracyStored = MeanStackedAccuracy[4:]
                print(*LabelledAccuracySorted, sep="\n")
                #torch.set_printoptions(profile="default")
                print(MeanStackedAccuracy)
                print(cm_tracker_unmod.get_aggregate_metrics())
                
                
                mAP_score_regular = np.mean(AP_regular)
                print("mAP score regular {:.2f}".format(mAP_score_regular))

                    

        print(f'finished epoch {epoch} in {time.time()-epochTime}')
        epochTime = time.time()
    print(model)