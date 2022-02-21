# DiGN module
# 
# Last updated: Dec 29 2021

import numpy as np
import matplotlib.pyplot as plt
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from resnet import ResNet18, ResNet18Wide, ResNet18_64, ResNet18Wide_64
from densenet import DenseNet121
from imagenet_models.resnet import ResNet50
from noise2net import Res2Net
from augment_and_mix import augment_and_mix

import datetime
from tqdm import tqdm

from torch.distributions.uniform import Uniform
from torch.distributions.exponential import Exponential

# list of methods that employ regularization
list_reg = ['GN', 'DiGN', 'trades', 'AugMix']

class DiGN:
    def __init__(self, arch='resnet18', n_classes=10, dataset='cifar10'):
        if arch == 'resnet18':
            self.model = ResNet18(num_classes=n_classes)
        elif arch == 'resnet18wide':
            self.model = ResNet18Wide(num_classes=n_classes)
        elif arch == 'resnet18_64':
            self.model = ResNet18_64(num_classes=n_classes)
        elif arch == 'resnet18wide_64':
            self.model = ResNet18Wide_64(num_classes=n_classes)
        elif arch == 'densenet121':
            self.model = DenseNet121(num_classes=n_classes)
        elif arch == 'inceptionv3':
            self.model = InceptionV3(num_classes=n_classes)
        elif arch == 'resnet50':
            self.model = ResNet50(num_classes=n_classes)
        else:
            raise ValueError('Invalid architecture.')
        self.device = (torch.device('cuda') if torch.cuda.is_available else torch.device('gpu'))
        if torch.cuda.device_count() > 1:
            print("Using ", torch.cuda.device_count(), "GPUs")
            self.model = nn.DataParallel(self.model)
        print('device detected: ', self.device)
        self.model.to(device=self.device)
        if dataset in ['cifar10','cifar100']:
            self.mean = (0.4914, 0.4822, 0.4465)
            self.std  = (0.2023, 0.1994, 0.2010)
        elif dataset in ['tinyimagenet', 'imagenette']:
            self.mean = (0.485, 0.456, 0.406)
            self.std  = (0.229, 0.224, 0.225)
        else:
            raise ValueError('dataset not supported.')
        self.loss_fn = nn.CrossEntropyLoss()

    def train(self, train_loader, val_loader, n_epochs, eval_freq=5,
              learning_rate=0.1, weight_decay=5e-4, momentum=0.9, ep_decay=50, gamma=0.1,
              adv_norm='Linf', epsilon=8/255, K=7, alpha=2.5 * (8/255)/ 7,
              stddev=0.1, random_std=False,        # Unif - also used in RSE
              lam_rate=0.1, random_exp=False,      # Exp
              n_samples=2,                         # number of stochastic samples
              alpha_coef=1.0, lam1=3.0, lam2=3.0,
              conf_adapt=False,                    # confidence weighting adaptation
              masking=False, probm=0.2,            # Bernoulli masking
              noisenet_max_eps=0.10,               # eps value for Noise2Net
              nplanes=16,                          # hidden planes in Noise2Net architecture
              aug_ratio=0.75,                      # DeepAugment augmentation/clean batch ratio
              mode='standard', model_path='./models/checkpoint_', log_dir='./logs/final_'):
        print('Training mode:', mode)
        self.model.train()

        if random_exp:
            m = Exponential(torch.tensor([lam_rate]))
        elif random_std:
            m = Uniform(torch.tensor([0.0]), torch.tensor([stddev]))
            
        val_acc_log = []

        optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum,
                              nesterov=True)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=ep_decay, gamma=gamma)

        acc_eval_best = 0.0
        for epoch in range(n_epochs):
            loss_train = 0.0
            
            if mode=='DeepAugment':
                for xx, _ in train_loader:
                    batch_size = xx.shape[0]
                    break
                noise2net_batch_size = int(batch_size * aug_ratio)
                noise2net = Res2Net(epsilon=noisenet_max_eps, hidden_planes=nplanes, batch_size=noise2net_batch_size).train().cuda()

            for x, y in tqdm(train_loader):
                batch_size = x.shape[0]
                H, W = x.shape[2], x.shape[3]

                x = x.to(device=self.device)  # Bx3x32x32
                y = y.to(device=self.device)

                if mode == 'DiGN':
                    # original
                    outputs = self.model_n(x)
                    p_orig = F.softmax(outputs, dim=1) # B x K
                    
                    reg = 0.0
                    
                    if lam1>0:                    
                        # gaussian noise consistency
                        avgKL = 0.0
                        for n in range(n_samples):
                            # make copy of tensor x
                            xc = x.clone().detach()
                            
                            if random_std: # Uniform random variable sigma with max value stddev
                                rn = torch.rand(batch_size)
                                rn = rn.unsqueeze(dim=1)
                                rn = rn.unsqueeze(dim=1)
                                rn = rn.unsqueeze(dim=1)
                                stdn = stddev*rn.to(device=self.device)
                            elif random_exp: # Exponential random variable sigma with rate lam
                                rn = m.sample(sample_shape=(batch_size,))
                                rn = rn.unsqueeze(dim=1)
                                rn = rn.unsqueeze(dim=1)
                                stdn = rn.to(device=self.device)
                            else: # fixed standard deviation sigma
                                stdn = stddev
                            
                            if masking:
                                u = torch.rand_like(xc)
#                                 probs = torch.rand(batch_size)
#                                 mask = torch.cat([(u[i].unsqueeze(dim=0)<probs[i])*1.0 for i in range(batch_size)])
                                mask = (u < probm)*1.0
                                mask = mask.to(device=self.device)
                            else:
                                mask = torch.ones_like(xc).to(device=self.device)
                            rn = stdn * mask * torch.randn_like(xc).to(device=self.device)

                            # add noise on top of clean x examples
                            outputs_n = self.model_n(xc, rn)
                            p_n = F.softmax(outputs_n, dim=1)
                            KL_n = (p_orig * torch.log((p_orig + 1e-14) / (p_n + 1e-14))).sum(dim=1)
                            avgKL += KL_n
                        avgKL /= n_samples
                        if conf_adapt:
                            tcp_orig = p_orig.gather(1, y.view(-1,1)).view(-1)
                            wAvgKL   = (avgKL*tcp_orig).mean()
                        else:
                            wAvgKL   = avgKL.mean()
                        reg = lam1 * wAvgKL
   
                    if lam2>0:
                        # augmix
                        x_am1 = AugMix(x)
                        outputs_am1 = self.model_n(x_am1)
                        p_am1  = F.softmax(outputs_am1, dim=1)

                        # augmix
                        x_am2 = AugMix(x)
                        outputs_am2 = self.model_n(x_am2)
                        p_am2  = F.softmax(outputs_am2, dim=1)

                        # Jensen-Shannon divergence
                        p_mix = (p_orig + p_am1 + p_am2)/3.0
                        KL_orig = torch.sum( p_orig*torch.log( (p_orig+1e-14)/(p_mix + 1e-14) ), dim=1).mean()
                        KL_am1  = torch.sum( p_am1*torch.log( (p_am1+1e-14)/(p_mix + 1e-14) ), dim=1).mean()
                        KL_am2  = torch.sum( p_am2*torch.log( (p_am2+1e-14)/(p_mix + 1e-14) ), dim=1).mean()
                        JS = (KL_orig + KL_am1 + KL_am2)/3.0
                    
                        reg += lam2 * JS
                elif mode == 'GN': # DiGN w.o. CR (+ AugMix)
                    # gaussian noise
                    if random_std:
                        r = torch.rand(batch_size)
                        r = r.unsqueeze(dim=1)
                        r = r.unsqueeze(dim=1)
                        r = r.unsqueeze(dim=1)
                        std = stddev*r.to(device=self.device)
                    else:
                        std = stddev
                    r = std * torch.randn_like(x).to(device=self.device)
                    outputs = self.model_n(x, r)
                    reg = 0
                    
                    if lam2>0:
                        outputs_orig = self.model_n(x)
                        p_orig  = F.softmax(outputs_orig, dim=1)
                        
                        # augmix
                        x_am1 = AugMix(x)
                        outputs_am1 = self.model_n(x_am1)
                        p_am1  = F.softmax(outputs_am1, dim=1)

                        # augmix
                        x_am2 = AugMix(x)
                        outputs_am2 = self.model_n(x_am2)
                        p_am2  = F.softmax(outputs_am2, dim=1)

                        # Jensen-Shannon divergence
                        p_mix = (p_orig + p_am1 + p_am2)/3.0
                        KL_orig = torch.sum( p_orig*torch.log( (p_orig+1e-14)/(p_mix + 1e-14) ), dim=1).mean()
                        KL_am1  = torch.sum( p_am1*torch.log( (p_am1+1e-14)/(p_mix + 1e-14) ), dim=1).mean()
                        KL_am2  = torch.sum( p_am2*torch.log( (p_am2+1e-14)/(p_mix + 1e-14) ), dim=1).mean()
                        JS = (KL_orig + KL_am1 + KL_am2)/3.0
                    
                        reg += lam2 * JS                
                elif mode == 'AugMix':
                    # original
                    outputs = self.model_n(x)
                    p_orig = F.softmax(outputs, dim=1)

                    # augmix 
                    x_am1 = AugMix(x)
                    outputs_am1 = self.model_n(x_am1)
                    p_am1  = F.softmax(outputs_am1, dim=1)

                    # augmix 
                    x_am2 = AugMix(x)
                    outputs_am2 = self.model_n(x_am2)
                    p_am2  = F.softmax(outputs_am2, dim=1)

                    # Jensen-Shannon divergence
                    p_mix = (p_orig + p_am1 + p_am2)/3.0
                    KL_orig = torch.sum( p_orig*torch.log( (p_orig+1e-14)/(p_mix + 1e-14) ), dim=1).mean()
                    KL_am1  = torch.sum( p_am1*torch.log( (p_am1+1e-14)/(p_mix + 1e-14) ), dim=1).mean()
                    KL_am2  = torch.sum( p_am2*torch.log( (p_am2+1e-14)/(p_mix + 1e-14) ), dim=1).mean()
                    JS = (KL_orig + KL_am1 + KL_am2)/3.0
                    reg = lam1*JS
                elif mode == 'trades':  # TRADES
                    d_adv = self.find_adv_input_KL(x, epsilon, K, alpha, adv_norm)
                    self.model.train()

                    outputs = self.model_n(x)
                    outputs_adv = self.model_n(x + d_adv)

                    # convert logits to softmax scores
                    p_orig = F.softmax(outputs, dim=1)
                    p_adv  = F.softmax(outputs_adv, dim=1)

                    KL = torch.sum( p_orig*torch.log( (p_orig+1e-14)/(p_adv + 1e-14) ), dim=1).mean()
                    reg = lam1*KL
                elif mode == 'adv':  # adversarial input perturbation
                    d_adv = self.find_adv_input(x, y, epsilon, K, alpha, adv_norm)
                    self.model.train()
                    outputs = self.model_n(x + d_adv)
                elif mode == 'rse': # random input perturbation
                    r = stddev * torch.randn_like(x).to(device=self.device)
                    outputs = self.model_n(x, r)
                elif mode == 'DeepAugment':
                    xc = x.clone().detach()
                    xc = normalize(xc, self.mean, self.std)
                    
                    with torch.no_grad():
                        # Setup network
                        noise2net.reload_parameters()
                        noise2net.set_epsilon(random.uniform(noisenet_max_eps / 2.0, noisenet_max_eps))

                        # Apply aug
                        x_auged = xc[:noise2net_batch_size].reshape((1, noise2net_batch_size * 3, H, W))
                        x_auged = noise2net(x_auged)
                        x_auged = x_auged.reshape((noise2net_batch_size, 3, H, W))
                        xc[:noise2net_batch_size] = x_auged
                    
                    xc = inv_normalize(xc, self.mean, self.std)
#                     xc = torch.clamp(xc, 0, 1)
                    
                    outputs = self.model_n(xc)
                elif mode == 'standard':  # standard
                    outputs = self.model_n(x)
                else:
                    raise NameError('Invalid mode')

                loss = self.loss_fn(outputs, y)
                if mode in list_reg:
                    loss += reg

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_train += loss.item()
            print(
                '{} Epoch {}, Train loss {}'.format(datetime.datetime.now(), epoch + 1, loss_train / len(train_loader)))
            scheduler.step()
            if (epoch == 0) or ((epoch + 1) % eval_freq == 0):
                acc_eval, _ = self.evaluate(val_loader)
                self.model.train()
                self.save_model(model_path, epoch + 1)
                val_acc_log.append((epoch, acc_eval))
                self.save_log(val_acc_log, log_dir)
                if acc_eval_best < acc_eval:
                    acc_eval_best = acc_eval
                    self.save_model(model_path + 'best')
        self.save_log(val_acc_log, log_dir)
        print('Training complete, model saved:', model_path)

    def model_n(self, x, r=0.0):
        """ Input normalization and model evaluation. Optional noise (r) is added post-normalization """
        xn = normalize(x, self.mean, self.std)
        outputs = self.model(xn + r)
        return outputs

    def find_adv_input_KL(self, x, epsilon, K, alpha, adv_norm='Linf'):
        """ Find input adversarial perturbation for KL distance
            x = input tensor
            y = label tensor
            epsilon = perturbation size
            K = number of steps
            alpha = step size
            adv_norm = L2 / Linf
        """
        self.model.eval()
        delta = torch.zeros_like(x, requires_grad=True).to(device=self.device)
        outputs = self.model_n(x)
        p_orig = F.softmax(outputs, dim=1).detach()
        for _ in range(K):
            # forward pass
            outputs_adv = self.model_n(x + delta)
            p_adv = F.softmax(outputs_adv, dim=1)
            cost = torch.sum(p_orig * torch.log((p_orig + 1e-14) / (p_adv + 1e-14)), dim=1).mean()

            # backward pass
            cost.backward()

            if adv_norm == 'L2':
                delta.data += alpha * delta.grad.detach() / norms(delta.grad.detach())
                delta.data *= epsilon / norms(delta.detach()).clamp(min=epsilon)
            else:  # adv_norm='Linf'
                delta.data = (delta + alpha * delta.grad.detach().sign()).clamp(-epsilon, epsilon)
            delta.grad.zero_()
        return delta.detach()

    def find_adv_input(self, x, y, epsilon, K, alpha, adv_norm='Linf'):
        """ Find input adversarial perturbation
            x = input tensor
            y = label tensor
            epsilon = perturbation size
            K = number of steps
            alpha = step size
            adv_norm = L2 / Linf
        """
        self.model.eval()
        delta = torch.zeros_like(x, requires_grad=True).to(device=self.device)
        for _ in range(K):
            # forward pass
            output = self.model_n(x + delta)
            cost = self.loss_fn(output, y)

            # backward pass
            cost.backward()

            if adv_norm == 'L2':
                delta.data += alpha * delta.grad.detach() / norms(delta.grad.detach())
                delta.data *= epsilon / norms(delta.detach()).clamp(min=epsilon)
            else:  # adv_norm='Linf'
                delta.data = (delta + alpha * delta.grad.detach().sign()).clamp(-epsilon, epsilon)
            delta.grad.zero_()
        return delta.detach()

    def evaluate(self, val_loader, ensemble=False, stddev=0.01, N=10, adv=False, epsilon=8/255.0, adv_norm='Linf', K=7):
        self.model.eval()
        correct = 0
        total = 0
        loss_eval = 0.0
        with torch.no_grad():
            for x, y in tqdm(val_loader):
                batch_size = x.shape[0]
                x = x.to(device=self.device)
                y = y.to(device=self.device)
                # construct adversarial attacks
                if adv:
                    with torch.enable_grad():
                        d_att = self.find_adv_input(x, y, epsilon=epsilon, K=K, alpha=2.5*epsilon/K, adv_norm=adv_norm)
                    xadv = x + d_att
                else:
                    xadv = x
                if ensemble:
                    for n in range(N):
                        delta = stddev * torch.randn_like(xadv).to(device=self.device)
                        if n == 0:
                            outputs = self.model_n(xadv+delta)
                        else:
                            outputs += self.model_n(xadv+delta)
                    outputs /= N
                else:  # standard eval
                    outputs = self.model_n(xadv)
                loss = self.loss_fn(outputs, y)
                _, yp = torch.max(outputs, dim=1)
                total += y.shape[0]
                correct += int((yp == y).sum())
                loss_eval += loss.item()
        acc = correct / total
        print(' Val acc = %.3f  Val loss %.5f' % (acc * 100.0, loss_eval / len(val_loader)))
        return acc, loss_eval / len(val_loader)
    
    def predict(self, val_loader, ensemble=False, stddev=0.01, N=10, adv=False, epsilon=8/255.0, K=7, n_classes=10):
        self.model.eval()
        total = 0
        softmax = torch.nn.Softmax(dim=1)
        p_pred = torch.zeros((len(val_loader.dataset),n_classes))
        y_true = torch.zeros((len(val_loader.dataset)))
        with torch.no_grad():
            start = 0
            for x, y in tqdm(val_loader):
                batch_size = x.shape[0]
                x = x.to(device=self.device)
                y = y.to(device=self.device)
                # construct adversarial attacks
                if adv:
                    with torch.enable_grad():
                        d_att = self.find_adv_input(x, y, epsilon=epsilon, K=K, alpha=2.5*epsilon/K, adv_norm='Linf')
                    xadv = x + d_att
                else:
                    xadv = x
                if ensemble:
                    for n in range(N):
                        delta = stddev * torch.randn_like(xadv).to(device=self.device)
                        if n == 0:
                            outputs = self.model_n(xadv+delta)
                        else:
                            outputs += self.model_n(xadv+delta)
                    outputs /= N
                else:  # standard eval
                    outputs = self.model_n(xadv)
                # convert logits to softmax scores
                p_pred[start:start+batch_size,:] = softmax(outputs)
                y_true[start:start+batch_size] = y
                total += y.shape[0]
                start += batch_size
        print('total examples = %d' % (total))
        return p_pred, y_true
    

    def save_model(self, save_path, ep=None):
        if ep is None:
            full_save_path = save_path + '.pt'
        else:
            full_save_path = save_path + str(ep) + '.pt'
        torch.save(self.model.state_dict(), full_save_path)

    def load_model(self, model_path):
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))

    def save_log(self, val_acc_log, dir):
        val_acc_log = np.array(val_acc_log)  # Nepochs x 2 (epoch, val_acc)
        np.save(dir + 'val_acc_log.npy', val_acc_log)

    def vis_log(self, filename, fsize=(10, 10)):
        la = np.load(filename)
        epochs, val_acc = la[:, 0], la[:, 1]
        plt.figure(figsize=fsize)
        plt.plot(epochs, val_acc)
        plt.xlabel('Epoch')
        plt.ylabel('Val Acc')
        plt.grid()
        plt.show()

# Helper functions
def AugMix(x, width=3, depth=-1, alpha=1.0):
    """ Augmix data creation, multiple mixing chains each with varying number of primitive operations applied
        Inputs:
            x = batch_size x 3 x 32 x 32
    """
    n=x.shape[0]
    x_augmix = torch.zeros_like(x).cpu()
    for i in range(n):
        x_numpy_version = x[i].cpu().numpy()
        severity = np.random.randint(1,11)
        x_augmix_temp = augment_and_mix(x_numpy_version.transpose(1,2,0), severity=severity, width=width, depth=depth, alpha=alpha)
        x_augmix[i] = torch.from_numpy(x_augmix_temp.transpose(2,0,1))
    return x_augmix.cuda()


def normalize(tensor, mean, std):
    for t, m, s in zip(tensor, mean, std):
        t.sub_(m).div_(s)
    return tensor


def inv_normalize(tensor, mean, std):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor


def norms(Z):
    """Compute norms over all but the first dimension"""
    return Z.view(Z.shape[0], -1).norm(dim=1)[:, None, None, None]
