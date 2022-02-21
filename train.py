# Training script for multiple independent runs, datasets, architectures
#
# Last updated: Dec 30 2021

import sys
from DataLoad import *

from DiGN import DiGN

args = sys.argv[1:]
dataset       = args[0]
architecture  = args[1]
batch_size    = int(args[2])
train_alg     = args[3]

print('Dataset: %s | Architecture: %s | Batch size: %d | Training Algorithm: %s' % (dataset, architecture, batch_size, train_alg))

if dataset=='cifar10':
    data_path = './cifar10'
    n_classes = 10
    get_loaders = get_loaders_cifar10
    wd = 5e-4
elif dataset=='cifar100':
    data_path = './cifar100'
    n_classes = 100
    get_loaders = get_loaders_cifar100
    wd = 5e-4
elif dataset=='tinyimagenet':
    data_path = './tiny-imagenet-200'
    n_classes = 200
    get_loaders = get_loaders_tinyimagenet
    wd = 5e-4
elif dataset=='imagenette':
    data_path = './Imagenette'
    n_classes = 10
    get_loaders = get_loaders_imagenette
    wd = 20e-4
else:
    raise ValueError('dataset not supported.')
    
if architecture=='resnet18':
    arch = 'RN18'
elif architecture=='resnet18wide':
    arch = 'WRN18'
elif architecture=='resnet18_64':
    arch = 'RN18_64'
elif architecture=='resnet18wide_64':
    arch = 'RN18W_64'
elif architecture=='resnet50':
    arch = 'RN50'
elif architecture=='densenet121':
    arch = 'DN121'
elif architecture=='inceptionv3':
    arch = 'IncV3'
else:
    raise ValueError('architecture not supported.')

# data loader for training, eval
train_loader, val_loader = get_loaders(data_path=data_path,
                                       batch_size_train=batch_size, batch_size_val=batch_size, num_workers=16)
print('# train batches = ', len(train_loader), ', # val batches = ', len(val_loader))

# number of runs
M = 3
Nepochs = 150
Neval = 5

# Model Training
if train_alg=='standard':
    mode = 'standard'
    for m in range(M):
        dign = DiGN(architecture, n_classes=n_classes, dataset=dataset)
        dign.train(train_loader, val_loader, n_epochs=Nepochs, eval_freq=Neval,
                     mode=mode,
                     weight_decay=wd,
                     model_path='./models/'+arch+'_'+dataset+'_Standard_run'+str(m+1)+'_chkpt_',
                     log_dir='./logs/'+arch+'_'+dataset+'_Standard_run'+str(m+1)+'_')
elif train_alg=='adversarial_training':
    mode = 'adv'
    for m in range(M):
        dign = DiGN(architecture, n_classes=n_classes, dataset=dataset)
        dign.train(train_loader, val_loader, n_epochs=Nepochs, eval_freq=Neval,
                     mode=mode,
                     weight_decay=wd,
#                      epsilon=4/255, K=7, alpha=2.5*(4/255)/7,
                     epsilon=8/255, K=7, alpha=2.5*(8/255)/7,
                     model_path='./models/'+arch+'_'+dataset+'_AT_run'+str(m+1)+'_chkpt_',
                     log_dir='./logs/'+arch+'_'+dataset+'_AT_run'+str(m+1)+'_')
elif train_alg=='trades':
    mode = 'trades'
    for m in range(M):
        dign = DiGN(architecture, n_classes=n_classes, dataset=dataset)
        dign.train(train_loader, val_loader, n_epochs=Nepochs, eval_freq=Neval,
                     mode=mode,
                     weight_decay=wd,
#                      epsilon=4/255, K=7, alpha=2.5*(4/255)/7,
                     epsilon=8/255, K=7, alpha=2.5*(8/255)/7,
                     model_path='./models/'+arch+'_'+dataset+'_Trades_run'+str(m+1)+'_chkpt_',
                     log_dir='./logs/'+arch+'_'+dataset+'_Trades_run'+str(m+1)+'_')
elif train_alg=='random_self_ensemble':
    mode = 'rse'
    for m in range(M):
        dign = DiGN(architecture, n_classes=n_classes, dataset=dataset)
        dign.train(train_loader, val_loader, n_epochs=Nepochs, eval_freq=Neval,
                     mode=mode,
                     weight_decay=wd,
                     stddev=0.1, # CIFAR10/100
                     # stddev=0.3, # Tiny-ImageNet
                     model_path='./models/'+arch+'_'+dataset+'_RSE_run'+str(m+1)+'_chkpt_',
                     log_dir='./logs/'+arch+'_'+dataset+'_RSE_run'+str(m+1)+'_')
elif train_alg=='augmix':
    mode = 'AugMix'
    for m in range(M):
        dign = DiGN(architecture, n_classes=n_classes, dataset=dataset)
        dign.train(train_loader, val_loader, n_epochs=Nepochs, eval_freq=Neval,
                     mode=mode,
                     weight_decay=wd,
                     lam1=12,
                     model_path='./models/'+arch+'_'+dataset+'_AugMix_run'+str(m+1)+'_chkpt_',
                     log_dir='./logs/'+arch+'_'+dataset+'_AugMix_run'+str(m+1)+'_')
elif train_alg=='gaussian_noise':
    mode = 'GN'
    for m in range(M):
        dign = DiGN(architecture, n_classes=n_classes, dataset=dataset)
        dign.train(train_loader, val_loader, n_epochs=Nepochs, eval_freq=Neval,
                     weight_decay=wd,
                     mode=mode,
                     lam1=0,
                     lam2=0,
                     stddev=0.2, random_std=True,
                     # stddev=0.6, random_std=True,
                     model_path='./models/'+arch+'_'+dataset+'_GN_run'+str(m+1)+'_chkpt_',
                     log_dir='./logs/'+arch+'_'+dataset+'_GN_run'+str(m+1)+'_')
elif train_alg=='dign':
    mode = 'DiGN'
    for m in range(M):
        dign = DiGN(architecture, n_classes=n_classes, dataset=dataset)
        dign.train(train_loader, val_loader, n_epochs=Nepochs, eval_freq=Neval,
                     mode=mode,
                     weight_decay=wd,
                     lam1=0.2,    # GN regularization weight
                     lam2=12.0,    # AugMix weight
                     stddev=0.2, random_std=True,  # Cifar-10/100
#                      stddev=0.6, random_std=True,  # Tiny-ImageNet
                     masking=False,
                     n_samples=3,                 # number of stochastic draws
                     model_path='./models/'+arch+'_'+dataset+'_DiGN_run'+str(m+1)+'_chkpt_',
                     log_dir='./logs/'+arch+'_'+dataset+'_DiGN_run'+str(m+1)+'_')
elif train_alg=='dign_gn':
    mode = 'DiGN'
    mname = 'DiGN_model_X'
    for m in range(M):
        dign = DiGN(architecture, n_classes=n_classes, dataset=dataset)
        dign.train(train_loader, val_loader, n_epochs=Nepochs, eval_freq=Neval,
                     mode=mode,
                     weight_decay=wd,
#                      conf_adapt=False,
                     lam1=0.2,    # GN weight
                     lam2=0,      # AugMix weight
                     stddev=0.2, random_std=True,
#                      probm=0.05, masking=True,
                     masking=False,
                     n_samples=3,
                     model_path='./models/'+arch+'_'+dataset+'_'+mname+'_run'+str(m+1)+'_chkpt_',
                     log_dir='./logs/'+arch+'_'+dataset+'_'+mname+'_run'+str(m+1)+'_')
elif train_alg=='deepaugment':
    mode = 'DeepAugment'
    eps_max = 0.20
    nplanes = 2
    aug_ratio = 0.75
    for m in range(M):
        dign = DiGN(architecture, n_classes=n_classes, dataset=dataset)
        dign.train(train_loader, val_loader, n_epochs=Nepochs, eval_freq=Neval,
                     mode=mode,
                     weight_decay=wd,
                     noisenet_max_eps=eps_max, nplanes=nplanes, aug_ratio=aug_ratio,
                     model_path='./models/'+arch+'_'+dataset+'_DeepAugment_run'+str(m+1)+'_chkpt_',
                     log_dir='./logs/'+arch+'_'+dataset+'_DeepAugment_run'+str(m+1)+'_')
else:
    raise ValueError('training algorithm not supported.')
