
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import time
import math
import timm
import timm.optim
import mz

def getDataLoader(dataset):
    return torch.utils.data.DataLoader(dataset,
        batch_size = batch_size,
        shuffle=True,
        num_workers=2,
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



def stepAtThreshold(x, threshold, k=10, base=1000):
    x = torch.pow(base, k * (x - threshold))
    x = x / x.sum(dim=1, keepdim=True)
    return x

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

# gradient based boundary calculation
class getDecisionBoundary(nn.Module):
    def __init__(self, initial_threshold = 0.5, lr = 3e-4, threshold_min = 0.2, threshold_max = 0.8):
        super().__init__()
        self.initial_threshold = initial_threshold
        self.thresholdPerClass = None
        self.lr = lr
        self.threshold_min = threshold_min
        self.threshold_max = threshold_max
        
    def forward(self, preds, targs):
        if self.thresholdPerClass == None:
            classCount = preds.size(dim=1)
            currDevice = preds.device
            self.thresholdPerClass = torch.ones(classCount, device=currDevice, requires_grad=True).to(torch.float64) * self.initial_threshold
        
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
    batch_size = 64
    grad_acc_epochs = 8
    num_classes = len(train_ds.classes)
    weight_decay = 2e-2
    
    torch.set_printoptions(linewidth = 200, sci_mode = False)
    
    device = torch.device("cuda:0")
    #device = torch.device("mps")
    #device = torch.device("cpu")

    datasets = {'train':train_ds,'val':test_ds}
    dataloaders = {x: getDataLoader(datasets[x]) for x in datasets}

    """# important stuff"""

    #model = timm.create_model('resnet_10t', pretrained=False, num_classes = num_classes)
    model = mz.resnet20(num_classes=num_classes)

    #model = mz.ViT(dim=128, num_classes=num_classes, depth=6, drop_path=0.1, drop=0.1)


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
    
    boundaryCalculator = getDecisionBoundary(initial_threshold = 0.5, lr = 3e-3)
    
    cycleTime = time.time()
    epochTime = time.time()
    stepsPerPrintout = 50
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
            correct2 = 0
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
                    preds = outputs.softmax(dim=1)
                    samples += len(image)
                    correct += sum(outputs.argmax(dim=1) == labels)
                    predsModified = preds - boundaryCalculator(preds, labelsOnehot)
                    preds = torch.argmax(predsModified, dim=1)
                    correct2 += sum(preds == labels)
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
                    top1v2 = 100 * (correct2/(samples+1e-8))
                    if i % stepsPerPrintout == 0:
                        imagesPerSecond = (batch_size * stepsPerPrintout)/(time.time() - cycleTime)
                        cycleTime = time.time()
                        print('[%d/%d][%d/%d]\tLoss: %.4f\tImages/Second: %.4f\tmAP: %.2f\ttop-1: %.2f\ttop-1 v2: %.2f\tP4: %.2f\tP4: %.2f' % (epoch,
                            num_epochs,
                            i,
                            len(dataloaders[phase]),
                            loss,
                            imagesPerSecond,
                            accuracy,
                            top1,
                            top1v2,
                            multiAccuracy[:,8].mean() * 100,
                            ma_2[8] * 100
                        ))

            
            print(*list(zip(cm_tracker.get_full_metrics() * 100, boundaryCalculator.thresholdPerClass)), sep="\n")
            print(cm_tracker.get_aggregate_metrics())
            print(f'top-1: {100 * (correct/samples)}% top-1 v2: {100 * (correct2/samples)}%')
        print(f'finished epoch {epoch} in {time.time()-epochTime}')
        epochTime = time.time()