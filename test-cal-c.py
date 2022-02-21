# Test script for Uncertainty Calibration in presence of common corruptions
# Evaluate calibration on clean and corrupted data
# 
# Last updated: Dec 30 2021

import sys
import numpy as np
import torch
from torchvision import datasets, transforms
from DataLoad import *
from DiGN import DiGN

args = sys.argv[1:]
dataset       = args[0]
architecture  = args[1]
batch_size    = int(args[2])
ensemble_eval = (args[3]=="True")
train_alg     = args[4]
eval_noise    = (args[5]=="True")

print('Dataset: %s | Architecture: %s | Batch size: %d' % (dataset, architecture, batch_size))
print('Ensemble_eval: %s | Train alg: %s' % (ensemble_eval, train_alg))
print('Evaluate noise only: %s' % (eval_noise))

if dataset in ['cifar10','cifar100']:
    ensemble_stddev = 0.1 # CIFAR10/100
else:
    ensemble_stddev = 0.3 # Tiny-ImageNet

if dataset=='cifar10':
    data_path = './cifar10'
    n_classes = 10
    get_loaders = get_loaders_cifar10
    corrupt_path = './CIFAR-10-C/'
elif dataset=='cifar100':
    data_path = './cifar100'
    n_classes = 100
    get_loaders = get_loaders_cifar100
    corrupt_path = './CIFAR-100-C/'
elif dataset=='tinyimagenet':
    data_path = './tiny-imagenet-200'
    n_classes = 200
    get_loaders = get_loaders_tinyimagenet
    corrupt_path = './Tiny-ImageNet-C/'
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
elif architecture=='densenet121':
    arch = 'DN121'
elif architecture=='inceptionv3':
    arch = 'IncV3'
else:
    raise ValueError('architecture not supported.')

# data loader for training, eval
train_loader, val_loader = get_loaders(data_path=data_path,
                                       batch_size_train=batch_size, batch_size_val=batch_size, num_workers=4)
print('# train batches = ', len(train_loader), ', # val batches = ', len(val_loader))

# architecture
dign = DiGN(architecture, n_classes=n_classes, dataset=dataset)

# number of runs
M = 3

# ======== Auxiliary Functions ===========

def get_corrupt_loader_cifar(corruption_path_base):
    labels = np.load(corruption_path_base+'labels.npy')
    if eval_noise:
        corruption_list=['speckle_noise','impulse_noise','shot_noise']        
    else:
        corruption_list=['saturate','spatter','gaussian_blur','speckle_noise','jpeg_compression','pixelate','elastic_transform','contrast','brightness','fog','frost','snow','zoom_blur','motion_blur','glass_blur','defocus_blur','impulse_noise','shot_noise'] #,'gaussian_noise']
    corruption_list.sort()
    
    x_all = np.zeros((50000*len(corruption_list),3,32,32))
    labels_all = np.zeros((50000*len(corruption_list)))

    start = 0
    for i in range(len(corruption_list)):
        x_corruption_i = np.load(corruption_path_base+corruption_list[i]+'.npy')
        x_corruption_i = np.moveaxis(x_corruption_i, 3, 1)
        x_all[start:start+50000] = x_corruption_i
        labels_all[start:start+50000] = labels
        start += 50000

    corrupt_loader = get_loader_from_numpy(x_all, labels_all, batch_size=500)
    return corrupt_loader

def get_corrupt_loader_tinyimagenet(corruption_path_base):
    # 14 corruptions
    if eval_noise:
        corruption_list=['impulse_noise','shot_noise']        
    else:
        corruption_list=['brightness','contrast','defocus_blur','elastic_transform','fog','frost','glass_blur','impulse_noise','jpeg_compression','motion_blur','pixelate','shot_noise','snow','zoom_blur'] #,'gaussian_noise']
    corruption_list.sort()
    
    # construct list of datasets
    Datasets = []
    for i in range(len(corruption_list)):
        corruption = corruption_list[i]
        for j in range(5):
            path = corruption_path_base+'/'+corruption+'/'+str(j+1)+'/'
            dataset = datasets.ImageFolder(path, transform=TEST_TRANSFORMS_DEFAULT(64))
            Datasets.append(dataset)
            
    # concatenate datasets
    # from: https://discuss.pytorch.org/t/how-does-concatdataset-work/60083/2
    all_datasets = torch.utils.data.ConcatDataset(Datasets)
#     all_datasets = torch.utils.data.ConcatDataset([d for d in Datasets])

    # construct dataloader for all corruptions, levels
    corrupt_loader = torch.utils.data.DataLoader(all_datasets, batch_size=500, shuffle=False)
        
    return corrupt_loader

# Measure how well prediction scores match the actual likelihood of a correct prediction
def calibr_metrics(p_pred, y_true, thres=0.5):
    y_true = y_true.astype('int')
    n = p_pred.shape[0]
    Delta = 0.0666    # bin resolution
    nbins = np.ceil(1/Delta).astype('int')
    p_range = np.linspace(0,1,nbins+1)

    # compute max-prob scores
    p_max = np.max(p_pred, axis=1)
    y_pred = np.argmax(p_pred, axis=1)
    
    # for each bin, compute accuracy and confidence
    acc = []
    conf = []
    ECE = 0.0   # expected calibration error (ECE)
    RMSE = 0.0  # RMS calibration error (RMSE)
    OE = 0.0    # overconfidence error (OE)
    for m in range(nbins):
        p_low, p_high = p_range[m], p_range[m+1]
        idx_m = np.where((p_max>p_low) & (p_max<=p_high))[0]
        card_Bm = len(idx_m)
        if card_Bm>0:
            conf_m = np.mean(p_max[idx_m])
            acc_m = np.sum(y_true[idx_m]==y_pred[idx_m])/card_Bm
            acc.append(acc_m)
            conf.append(conf_m)
            ECE += card_Bm/n*np.abs(acc_m-conf_m)
            RMSE += card_Bm/n*((acc_m-conf_m)**2)
            OE += card_Bm/n*conf_m*np.max([conf_m-acc_m,0])
    conf = np.array(conf).reshape((len(conf),1))
    acc = np.array(acc).reshape((len(acc),1))
    RMSE = np.sqrt(RMSE)
    return ECE, RMSE, OE, acc, conf

def aggregate_cal_results(dataset, arch, train_alg):
    import pandas as pd

    # Load results
    ece1, rmse1, oe1, acc1, conf1, ece_cor1, rmse_cor1, oe_cor1, acc_cor1, conf_cor1 = np.load('./results/cal_'+arch+'_'+dataset+'_'+train_alg+'_run1.npy', allow_pickle=True)
    ece2, rmse2, oe2, acc2, conf2, ece_cor2, rmse_cor2, oe_cor2, acc_cor2, conf_cor2 = np.load('./results/cal_'+arch+'_'+dataset+'_'+train_alg+'_run2.npy', allow_pickle=True)
    ece3, rmse3, oe3, acc3, conf3, ece_cor3, rmse_cor3, oe_cor3, acc_cor3, conf_cor3 = np.load('./results/cal_'+arch+'_'+dataset+'_'+train_alg+'_run3.npy', allow_pickle=True)

    # Average RMSE calibration errors
    mean_clean_rmse, std_clean_rmse = np.mean([rmse1,rmse2,rmse3]), np.std([rmse1,rmse2,rmse3])
    mean_corr_rmse, std_corr_rmse   = np.mean([rmse_cor1,rmse_cor2,rmse_cor3]), np.std([rmse_cor1,rmse_cor2,rmse_cor3])
    
    # [Metric] [Mean] [Std]
    metrics = ['Clean RMSE', 'Corrupt RMSE']
    mean_array = [mean_clean_rmse, mean_corr_rmse]
    std_array = [std_clean_rmse, std_corr_rmse]
    
    data = {
        'Metric':metrics,
        'Mean':list(100*np.array(mean_array)),
        'Std':list(100*np.array(std_array))
    }

    results_df = pd.DataFrame(data)
    print(results_df)
    
    # save DataFrame
    results_df.to_pickle('./results/cal_'+arch+'_'+dataset+'_'+train_alg+'_'+'aggr_cal_'+str(eval_noise)+'.pkl')
    print('Dataframe saved at: ./results/cal_'+arch+'_'+dataset+'_'+train_alg+'_'+'aggr_cal_'+str(eval_noise)+'.pkl')


# get loader for corrupted data
if dataset in ['cifar10','cifar100']:
    get_corrupt_loader = get_corrupt_loader_cifar
else: # Tiny-ImageNet
    get_corrupt_loader = get_corrupt_loader_tinyimagenet
    
corrupt_loader = get_corrupt_loader(corrupt_path)

# For each training run, compute corruption results
for m in range(M):
    # load model
    model_name = './models/'+arch+'_'+dataset+'_'+train_alg+'_run'+str(m+1)+'_chkpt_best.pt'
    dign.load_model(model_name)
    print(model_name)

    p_pred, y_true = dign.predict(val_loader, ensemble=ensemble_eval, stddev=ensemble_stddev, N=10, n_classes=n_classes)
    p_pred = p_pred.detach().cpu().numpy()
    y_true = y_true.detach().cpu().numpy()
    ece, rmse, oe, acc, conf = calibr_metrics(p_pred, y_true)
    print("Clean: ECE = %.2f | RMSE = %.2f | OE = %.2f " % (ece*100, rmse*100, oe*100))

    p_pred, y_true = dign.predict(corrupt_loader, ensemble=ensemble_eval, stddev=ensemble_stddev, N=10, n_classes=n_classes)
    p_pred = p_pred.detach().cpu().numpy()
    y_true = y_true.detach().cpu().numpy()
    ece_cor, rmse_cor, oe_cor, acc_cor, conf_cor = calibr_metrics(p_pred, y_true)
    print("Corrupt: ECE = %.2f | RMSE = %.2f | OE = %.2f " % (ece_cor*100, rmse_cor*100, oe_cor*100))
    
    # Save results in .npy file
    np.save('./results/cal_'+arch+'_'+dataset+'_'+train_alg+'_run'+str(m+1)+'.npy', [ece, rmse, oe, acc, conf, ece_cor, rmse_cor, oe_cor, acc_cor, conf_cor])
    print('Results saved at: '+'./results/cal_'+arch+'_'+dataset+'_'+train_alg+'_run'+str(m+1)+'.npy')


# call aggregator for calibration results
aggregate_cal_results(dataset, arch, train_alg)
