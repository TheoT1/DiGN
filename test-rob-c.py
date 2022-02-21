# Test script for testing robustness against common corruptions
# Evaluate robustness on clean and corrupted data
# 
# Last updated: Dec 30 2021

import sys
import numpy as np
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
elif dataset=='imagenette':
    data_path = './Imagenette'
    n_classes = 10
    get_loaders = get_loaders_imagenette
    corrupt_path = './ImagenetteC/'    
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
                                       batch_size_train=batch_size, batch_size_val=batch_size, num_workers=4)
print('# train batches = ', len(train_loader), ', # val batches = ', len(val_loader))

# architecture
dign = DiGN(architecture, n_classes=n_classes, dataset=dataset)

# number of runs
M = 3

# eval against corruptions for CIFAR-10/100
def eval_corruptions_cifar(corruption_level, corruption_path_base):
    # corruption evaluation for solve set corruption level \in [1,2,3,4,5]
    print('Corruption level:', corruption_level)

    labels = np.load(corruption_path_base+'labels.npy')
    if eval_noise:
        corruption_list=['gaussian_noise','speckle_noise','impulse_noise','shot_noise']        
    else:
        corruption_list=['saturate','spatter','gaussian_blur','speckle_noise','jpeg_compression','pixelate','elastic_transform','contrast','brightness','fog','frost','snow','zoom_blur','motion_blur','glass_blur','defocus_blur','impulse_noise','shot_noise'] #,'gaussian_noise']
    corruption_list.sort()

    classification_accuracies = np.zeros((1,len(corruption_list)))
    for i in range(len(corruption_list)):        
        current_corruption = np.load(corruption_path_base+corruption_list[i]+'.npy')

        x_intermediate = current_corruption[10000*(corruption_level-1):10000*(corruption_level),:,:,:]
        x_final = np.moveaxis(x_intermediate, 3, 1)
        labels_section = labels[10000*(corruption_level-1):10000*(corruption_level)]

        my_dataloader = get_loader_from_numpy(x_final,labels_section,batch_size=500)
        if ensemble_eval==False:
            acc_eval, loss_eval = dign.evaluate(my_dataloader)
        else:
            acc_eval, loss_eval = dign.evaluate(my_dataloader,ensemble=True,stddev=ensemble_stddev,N=10)

        print('Current Corruption: '+corruption_list[i]+'--'+'Classification Accuracy: '+str(acc_eval)[:6])
        classification_accuracies[0,i] = acc_eval

    return corruption_list, classification_accuracies

# eval against corruptions for Tiny-ImageNet
def eval_corruptions_tinyimagenet(corruption_level, corruption_path_base):
    # corruption evaluation for solve set corruption level \in [1,2,3,4,5]
    print('Corruption level:', corruption_level)

    # 14 corruptions
    if eval_noise:
        corruption_list=['gaussian_noise','impulse_noise','shot_noise']        
    else:
        corruption_list=['brightness','contrast','defocus_blur','elastic_transform','fog','frost','glass_blur','impulse_noise','jpeg_compression','motion_blur','pixelate','shot_noise','snow','zoom_blur'] #,'gaussian_noise']
    corruption_list.sort()

    classification_accuracies = np.zeros((1,len(corruption_list)))
    for i in range(len(corruption_list)): # for each corruption
        # construct dataloader for (corruption,level)
        corruption = corruption_list[i]
        my_dataloader = get_loader_from_path(corruption_path_base+'/'+corruption+'/'+str(corruption_level)+'/', batch_size=500)
        if ensemble_eval==False:
            acc_eval, loss_eval = dign.evaluate(my_dataloader)
        else:
            acc_eval, loss_eval = dign.evaluate(my_dataloader,ensemble=True,stddev=ensemble_stddev,N=10)

        print('Current Corruption: '+corruption_list[i]+'--'+'Classification Accuracy: '+str(acc_eval)[:6])
        classification_accuracies[0,i] = acc_eval

    return corruption_list, classification_accuracies

# eval against corruptions for Imagenette
def eval_corruptions_imagenette(corruption_level, corruption_path_base):
    # corruption evaluation for solve set corruption level \in [1,2,3,4,5]
    print('Corruption level:', corruption_level)

    # 18 corruptions
    if eval_noise:
        corruption_list=['gaussian_noise','speckle_noise','impulse_noise','shot_noise']        
    else:
        corruption_list=['brightness','contrast','defocus_blur','elastic_transform','fog','frost','gaussian_blur','glass_blur','impulse_noise','jpeg_compression','motion_blur','pixelate','saturate','shot_noise','snow','spatter','speckle_noise','zoom_blur'] #,'gaussian_noise']
    corruption_list.sort()

    classification_accuracies = np.zeros((1,len(corruption_list)))
    for i in range(len(corruption_list)): # for each corruption
        # construct dataloader for (corruption,level)
        corruption = corruption_list[i]
        my_dataloader = get_loader_from_path(corruption_path_base+'/'+corruption+'/'+str(corruption_level)+'/', batch_size=500)
        if ensemble_eval==False:
            acc_eval, loss_eval = dign.evaluate(my_dataloader)
        else:
            acc_eval, loss_eval = dign.evaluate(my_dataloader,ensemble=True,stddev=ensemble_stddev,N=10)

        print('Current Corruption: '+corruption_list[i]+'--'+'Classification Accuracy: '+str(acc_eval)[:6])
        classification_accuracies[0,i] = acc_eval

    return corruption_list, classification_accuracies

# For each training run, compute corruption results
for m in range(M):
    # load model
    model_name = './models/'+arch+'_'+dataset+'_'+train_alg+'_run'+str(m+1)+'_chkpt_best.pt'

    dign.load_model(model_name)
    print(model_name)

    # clean evaluation
    if ensemble_eval==False:
        acc_eval, loss_eval = dign.evaluate(val_loader)
    else:
        acc_eval, loss_eval = dign.evaluate(val_loader,ensemble=True,stddev=ensemble_stddev,N=10)
    
    cleana = acc_eval
    
    if dataset in ['cifar10','cifar100']:
        eval_corruptions = eval_corruptions_cifar
    elif dataset=='tinyimagenet':
        eval_corruptions = eval_corruptions_tinyimagenet
    elif dataset=='imagenette':
        eval_corruptions = eval_corruptions_imagenette
    else:
        raise ValueError('invalid dataset')

    # Get classification accuracy at each severy level
    corruption_list, classification_accuracies1 = eval_corruptions(1,corrupt_path)
    corruption_list, classification_accuracies2 = eval_corruptions(2,corrupt_path)
    corruption_list, classification_accuracies3 = eval_corruptions(3,corrupt_path)
    corruption_list, classification_accuracies4 = eval_corruptions(4,corrupt_path)
    corruption_list, classification_accuracies5 = eval_corruptions(5,corrupt_path)

    classification_accuracies_avg = (1/5)*(classification_accuracies1+classification_accuracies2+classification_accuracies3+classification_accuracies4+classification_accuracies5)

    print('\n\nFinal Values: ')

    # mean
    print(np.array(corruption_list).T)
    print(np.array(classification_accuracies_avg).T)
    print(np.array(corruption_list).T.shape, np.array(classification_accuracies_avg).T.shape)
    mca = np.mean(classification_accuracies_avg[0])
    print('Mean Corrupt Accuracy: ', mca)

    corruption_list = list(corruption_list)
    acc_list = list(classification_accuracies_avg[0])
    for i in range(len(corruption_list)):
        print('%s \t\t| %.2f' % (corruption_list[i], acc_list[i]*100.0))
        
    # Save results in .npy file
    np.save('./results/'+arch+'_'+dataset+'_'+train_alg+'_run'+str(m+1)+'.npy', [cleana, mca, corruption_list, np.array(acc_list)])
    print('Results saved at: '+'./results/'+arch+'_'+dataset+'_'+train_alg+'_run'+str(m+1)+'.npy')


# ======== Auxiliary Functions ===========
def aggregate_rob_results(dataset, arch, train_alg):
    import pandas as pd

    # Load results
    cleana1, mca1, corruption_list, acc_list1 = np.load('./results/'+arch+'_'+dataset+'_'+train_alg+'_run1.npy', allow_pickle=True)
    cleana2, mca2, corruption_list, acc_list2 = np.load('./results/'+arch+'_'+dataset+'_'+train_alg+'_run2.npy', allow_pickle=True)
    cleana3, mca3, corruption_list, acc_list3 = np.load('./results/'+arch+'_'+dataset+'_'+train_alg+'_run3.npy', allow_pickle=True)

    # Average accuracies
    mean_clean_acc, std_clean_acc = np.mean([cleana1,cleana2,cleana3]), np.std([cleana1,cleana2,cleana3])
    mean_mca, std_mca = np.mean([mca1,mca2,mca3]), np.std([mca1,mca2,mca3])
    mean_ca = np.mean([acc_list1, acc_list2, acc_list3], axis=0)
    std_ca  = np.std([acc_list1, acc_list2, acc_list3], axis=0)

    # [Metric] [Mean acc] [Std acc]
    metrics = ['Clean','mCA']
    metrics += corruption_list
    mean_acc_array = [mean_clean_acc,mean_mca]
    mean_acc_array += list(mean_ca)
    std_acc_array = [std_clean_acc,std_mca]
    std_acc_array += list(std_ca)

    data = {
        'Metric':metrics,
        'Mean Accuracy':list(100*np.array(mean_acc_array)),
        'Std Accuracy':list(100*np.array(std_acc_array))
    }

    corruptions_df = pd.DataFrame(data)
    corruptions_df.head(22)
    print(corruptions_df)
    
    # save DataFrame
    corruptions_df.to_pickle('./results/'+arch+'_'+dataset+'_'+train_alg+'_'+'aggr_rob_'+str(eval_noise)+'.pkl')
    print('Dataframe saved at: ./results/'+arch+'_'+dataset+'_'+train_alg+'_'+'aggr_rob_'+str(eval_noise)+'.pkl')

# call aggregator
aggregate_rob_results(dataset, arch, train_alg)
