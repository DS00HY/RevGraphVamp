import torch
import numpy as np
import deeptime
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import mdshare
from torch.utils.data import DataLoader
import json
from args import buildParser
from layers import GaussianDistance, NeighborMultiHeadAttention, InteractionBlock, GraphConvLayer
from model import GraphVampNet
from revvamp import *
from deeptime.util.data import TrajectoryDataset
from deeptime.decomposition.deep import VAMPNet
from deeptime.decomposition import VAMP
from copy import deepcopy
import os
import pickle
import warnings
#from deeptime.decomposition._koopman import  KoopmanChapmanKolmogorovValidator
from utils_vamp import *

if torch.cuda.is_available():
    device = torch.device('cuda')
    print('cuda is available')
else:
    print('Using CPU')
    device = torch.device('cpu')

# ignore deprecation warnings
warnings.filterwarnings('ignore',category=DeprecationWarning)

args = buildParser().parse_args()

def record_result(string, file_name):
    if not os.path.exists(file_name):
        with open(file_name, 'w') as f:
            print("successfully create record file!")
            f.write(string + "\n")
    with open(file_name, 'a') as f:
        f.write(string+"\n")

log_pth = f'{args.save_folder}/training.log'

if not os.path.exists(args.save_folder):
    print('making the folder for saving checkpoints')
    os.makedirs(args.save_folder)

with open(args.save_folder+'/args.txt','w') as f:
    f.write(str(args))

meta_file = os.path.join(args.save_folder, 'metadata.pkl')
pickle.dump({'args': args}, open(meta_file, 'wb'))

#------------------- data as a list of trajectories ---------------------------
file_path = args.data_path
data_info_file = file_path + "datainfo.npy"
dist_file = file_path + "dist.npy"
nbr_data_file = file_path + "inds.npy"

#data_info = np.load("../intermediate/red_5nbrs_1ns_datainfo.npy", allow_pickle=True).item()
data_info = np.load(args.data_info, allow_pickle=True).item()
#data_info = np.load("../intermediate/red_5nbrs_1ns_datainfo.npy", allow_pickle=True).item()
traj_length = data_info['length']
print(traj_length)
print(traj_length[0].shape)
#dists1, inds1 = np.load(args.dist_data)['arr_0'], np.load(args.nbr_data)['arr_0']
dists1, inds1 = np.load(args.dist_data), np.load(args.nbr_data)
print(dists1.shape)
print(inds1.shape)

#
# #dt = 0.25  # Trajectory timestep in ns
# #generator = DataGenerator(input_data[k], ratio=args.val_frac, dt=dt, max_frames=1000000)
#
dist_sp= [r for r in unflatten(dists1, traj_length)]
inds_sp = [r for r in unflatten(inds1, traj_length)]
data = []
for i in range(len(dist_sp)):
    mydists1 = torch.from_numpy(dist_sp[i])
    myinds1 = torch.from_numpy(inds_sp[i])
    data.append(torch.cat((mydists1, myinds1), axis=-1))


dataset = TrajectoryDataset.from_trajectories(lagtime=args.tau, data=data)

n_val = int(len(dataset)*args.val_frac)
train_data, val_data = torch.utils.data.random_split(dataset, [len(dataset)-n_val, n_val])

loader_train = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
loader_val = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)
# torch.save(loader_train, 'abtrainala.pth')
# torch.save(loader_val, 'abvalala.pth')
# loader_train = torch.load('abtrainala.pth')
# loader_val = torch.load('abvalala.pth')
#
# torch.save(loader_train, 'ab_mintrainala.pth')
# torch.save(loader_val, 'ab_minvalala.pth')
# loader_train = torch.load('ab_mintrainala.pth')
# loader_val = torch.load('ab_minvalala.pth')

# torch.save(loader_train, 'ab_mintrainala_2.pth')
# torch.save(loader_val, 'ab_minvalala_2.pth')
# loader_train = torch.load('ab_mintrainala_2.pth')
# loader_val = torch.load('ab_minvalala_2.pth')

all_batch = len(dataset)
if all_batch > 250000 or args.num_atoms > 10:
    all_batch = int(all_batch)

loader_train_all = DataLoader(train_data, batch_size=all_batch, shuffle=True)
print("data size=", len(dataset))
# data is a list of trajectories [T,N,M+M]
#---------------------------------------------------------------------------------

lobe = GraphVampNet()
#print(lobe)

lobe_timelagged = deepcopy(lobe).to(device=device)
lobe = lobe.to(device)
print(args.score_method)
vlu = vls = None
vlu = VAMPU(args.num_classes, activation=torch.exp)
vls = VAMPS(args.num_classes, activation=torch.exp, renorm=True)

vampnet = RevVAMPNet(lobe=lobe, lobe_timelagged=lobe_timelagged, learning_rate=args.lr, device=device, optimizer='Adam',
                     score_method=args.score_method, vampu=vlu, vamps=vls)

def count_parameters(model):
    '''
    count the number of parameters in the model
    '''
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print('number of parameters', count_parameters(lobe))

def train(train_loader , n_epochs, validation_loader=None, loader_train_all=None):
    '''
    Parameters:
    -----------------
    train_loader: torch.utils.data.DataLoader
        The data to use for training, should yield a tuple of batches representing instantaneous and time-lagged samples
    n_epochs: int, the number of epochs for training
    validation_loader: torch.utils.data.DataLoader:
        The validation data should also be yielded as a two-element tuple.

    Returns:
    -----------------
    model: VAMPNet
    '''

    n_OOM = 0
    pre_epoch = args.pre_train_epoch

    if args.score_method == 'VAMPCE':

        print("Train the vanilla VAMPNet model.")
        vampnet.vampu.requires_grad_(False)
        vampnet.vamps.requires_grad_(False)
        vampnet.score_method = 'VAMP2'
        print(vampnet.score_method)
        early_stopping = EarlyStopping(args.save_folder, file_name='best_pre_lobe',delta=1e-4, patience=100)
        best_dict = None
        for epoch in tqdm(range(pre_epoch)):
            try:
                for batch_0, batch_t in train_loader:
                    torch.cuda.empty_cache()  # 释放显存
                    vampnet.partial_fit((batch_0.to(device), batch_t.to(device)))
                if validation_loader is not None:
                    with torch.no_grad():
                        scores = []
                        for val_batch in validation_loader:
                            scores.append(vampnet.validate((val_batch[0].to(device), val_batch[1].to(device))))
                        mean_score = torch.mean(torch.stack(scores))
                    early_stopping(mean_score.item(), {'state_dict': vampnet.lobe.state_dict()})
                    if early_stopping.is_best:
                        best_dict = vampnet.lobe.state_dict()
                    if early_stopping.early_stop:
                        print("Early stopping pre lobe")
                        break
                if epoch % 10 == 9:
                    record_result("pre lobe train step: %f, vill score %s , max = %s" % (epoch, mean_score.item(),
                                                                                         early_stopping.val_loss_min), log_pth)
            except RuntimeError as e:
                print(epoch, "run error !!!!!")
                print(e, e.args[0])
                n_OOM += 1

        vampnet.vampu.requires_grad_(True)
        vampnet.vamps.requires_grad_(True)
        vampnet.score_method = 'VAMPCE'
        if best_dict:
            vampnet.lobe.load_state_dict(best_dict)
            #vampnet.lobe_timelagged = deepcopy(lobe).to(device=device)
            vampnet.lobe_timelagged.load_state_dict(deepcopy(best_dict))
        print("vamp2 max score", (early_stopping.val_loss_min))

        # modelpath =  args.save_folder + '/best_pre_lobe.pt'
        # checkpoint = torch.load(modelpath)
        # vampnet.lobe.load_state_dict(checkpoint['state_dict'])
        # vampnet.lobe_timelagged = deepcopy(vampnet.lobe).to(device=device)
        # vampnet.lobe = vampnet.lobe.to(device)


        print("Training auxiliary network...")
        #vampnet.set_optimizer_lr(1e-1) # reduce learning rate
        early_stopping = EarlyStopping(args.save_folder, file_name='best_pre', delta=1e-4, patience=100)
        best_dict = None
        vampnet.lobe.requires_grad_(False)
        vampnet.lobe_timelagged.requires_grad_(False)

        # change vls,vlu weight
        data_size = 0
        for batch_0, batch_t in train_loader:
            data_size += batch_0.shape[0]
        state_probs = np.zeros((data_size, int(args.num_classes)))
        state_probs_tau = np.zeros((data_size, int(args.num_classes)))
        n_iter = 0
        with torch.no_grad():
            for batch_0, batch_t in train_loader:
                torch.cuda.empty_cache()
                batch_size = len(batch_0)
                # print(batch_size)
                state_probs[n_iter:n_iter + batch_size] = vampnet.transform(batch_0)
                state_probs_tau[n_iter:n_iter + batch_size] = vampnet.transform(batch_t, instantaneous=False)
                n_iter += batch_size
            vampnet.update_auxiliary_weights([state_probs, state_probs_tau], optimize_S=True)

        # train auxiliary network
        for epoch in tqdm(range(pre_epoch)):
            try:

                torch.cuda.empty_cache()
                vampnet.train_US([state_probs,state_probs_tau])

                if validation_loader is not None:
                    with torch.no_grad():
                        scores = []
                        for val_batch in validation_loader:
                            scores.append(vampnet.validate((val_batch[0].to(device), val_batch[1].to(device))))
                        mean_score = torch.mean(torch.stack(scores))
                    early_stopping(mean_score.item(), {
                                                        'epoch': pre_epoch,
                                                        'state_dict': lobe.state_dict(),
                                                        'vlu_dict': vlu.state_dict(),
                                                        'vls_dict': vls.state_dict(),
                                                    })

                    if epoch % 10 == 9:
                        record_result("pre us train step: %f, vill score %s " % (epoch, mean_score.item()), log_pth)
                    if early_stopping.is_best:
                        best_dict = {'vlu_dict': vlu.state_dict(), 'vls_dict': vls.state_dict()}
                    if early_stopping.early_stop:
                        print("Early stopping pre U S")
                        break
            except RuntimeError as e:
                print(epoch, "run error !!!!!")
                print(e, e.args[0])
                n_OOM += 1
        del state_probs
        del state_probs_tau
        print("now vampce score", (early_stopping.val_loss_min))
        vampnet.lobe.requires_grad_(True)
        vampnet.lobe_timelagged.requires_grad_(True)
        if best_dict:
            vampnet.vampu.load_state_dict(best_dict['vlu_dict'])
            vampnet.vamps.load_state_dict(best_dict['vls_dict'])
        # checkpoint = torch.load(modelpath)
        # vampnet.lobe.load_state_dict(checkpoint['state_dict'])
        # vampnet.lobe_timelagged = deepcopy(vampnet.lobe).to(device=device)
        # vampnet.lobe = vampnet.lobe.to(device)
        # vampnet.vampu.load_state_dict(checkpoint['vlu_dict'])
        # vampnet.vamps.load_state_dict(checkpoint['vls_dict'])
        record_result("pretrain step: %f, vill score %s " % (pre_epoch, early_stopping.val_loss_min), log_pth)
        vampnet.set_optimizer_lr(0.2)  # reduce learning rate




    print("all network...")
    all_train_epo = 0
    best_score = 0
    best_epoch = 0
    best_model = None
    early_stopping = EarlyStopping(args.save_folder, file_name='best_allnet', delta=1e-4, patience=200)
    for epoch in tqdm(range(n_epochs)):
        '''
        perform batches of data here
        '''
        if epoch == 100:
            vampnet.set_optimizer_lr(0.2)  # reduce learning rate
        try:
            now_train_num = 0
            for batch_0, batch_t in train_loader:
                torch.cuda.empty_cache()
                vampnet.partial_fit((batch_0.to(device), batch_t.to(device)))
                all_train_epo +=1
                now_train_num +=1

            if validation_loader is not None:
                with torch.no_grad():
                    scores = []
                    for val_batch in validation_loader:
                        scores.append(vampnet.validate((val_batch[0].to(device), val_batch[1].to(device))))

                    mean_score = torch.mean(torch.stack(scores))
                    vampnet._validation_scores.append((vampnet._step, mean_score.item()))
                early_stopping(mean_score.item(), {
                    'epoch': pre_epoch,
                    'state_dict': lobe.state_dict(),
                    'vlu_dict': vlu.state_dict(),
                    'vls_dict': vls.state_dict(),
                })
                if early_stopping.is_best:
                    best_model = {'epoch': pre_epoch, 'lobe': lobe.state_dict(), 'vlu': vlu.state_dict(), 'vls': vls.state_dict()}
                    best_epoch = epoch
                if early_stopping.early_stop:
                    print("Early stopping all network train")
                    break

            if epoch % 10 == 9 and args.save_checkpoints:
                if epoch % 50 == 9:
                    torch.save({
                        'epoch' : epoch,
                        'state_dict': lobe.state_dict(),
                        'vlu_dict' : vlu.state_dict(),
                        'vls_dict': vls.state_dict(),
                        }, args.save_folder+'/logs_'+str(epoch)+'.pt')
                record_result("step: %f, mean_sroce %f , trainmean %f , max= %f" % (epoch, mean_score.item(),
                                   np.mean(vampnet.train_scores[-now_train_num-1:-1][0,1]), early_stopping.val_loss_min), log_pth)
        except RuntimeError as e:
            print(epoch, "run error !!!!!")
            print(e, e.args[0])
            n_OOM += 1

    if best_model:
        print("best model score is  ", early_stopping.val_loss_min)
        vampnet.lobe.load_state_dict(best_model['lobe'])
        vampnet.lobe_timelagged.load_state_dict(deepcopy(best_model['lobe']))
        #vampnet.lobe_timelagged = deepcopy(lobe).to(device=device)
        vampnet.vampu.load_state_dict(best_model['vlu'])
        vampnet.vamps.load_state_dict(best_model['vls'])
    return vampnet.fetch_model(), all_train_epo


def train_US(train_loader , n_epochs, validation_loader=None, loader_train_all=None, model_name="beat_us"):
    n_OOM = 0
    pre_epoch = n_epochs
    print("Training auxiliary network...")
    # vampnet.set_optimizer_lr(1e-1) # reduce learning rate
    early_stopping = EarlyStopping(args.save_folder, file_name=model_name, delta=1e-4, patience=50)
    best_dict = None
    vampnet.lobe.requires_grad_(False)
    vampnet.lobe_timelagged.requires_grad_(False)

    # change vls,vlu weight
    data_size = 0
    for batch_0, batch_t in train_loader:
        data_size += batch_0.shape[0]
    state_probs = np.zeros((data_size, int(args.num_classes)))
    state_probs_tau = np.zeros((data_size, int(args.num_classes)))
    n_iter = 0
    with torch.no_grad():
        for batch_0, batch_t in train_loader:
            torch.cuda.empty_cache()
            batch_size = len(batch_0)
            # print(batch_size)
            state_probs[n_iter:n_iter + batch_size] = vampnet.transform(batch_0)
            state_probs_tau[n_iter:n_iter + batch_size] = vampnet.transform(batch_t, instantaneous=False)
            n_iter += batch_size
        vampnet.update_auxiliary_weights([state_probs, state_probs_tau], optimize_S=True)

    # train auxiliary network
    for epoch in tqdm(range(n_epochs)):
        try:

            torch.cuda.empty_cache()
            vampnet.train_US([state_probs, state_probs_tau])

            if validation_loader is not None:
                with torch.no_grad():
                    scores = []
                    for val_batch in validation_loader:
                        scores.append(vampnet.validate((val_batch[0].to(device), val_batch[1].to(device))))
                    mean_score = torch.mean(torch.stack(scores))
                early_stopping(mean_score.item(), {
                    'epoch': pre_epoch,
                    'state_dict': lobe.state_dict(),
                    'vlu_dict': vlu.state_dict(),
                    'vls_dict': vls.state_dict(),
                })

                if epoch % 50 == 9:
                    record_result("pre us train step: %f, vill score %s " % (epoch, mean_score.item()), log_pth)
                if early_stopping.is_best:
                    best_dict = {'vlu_dict': vlu.state_dict(), 'vls_dict': vls.state_dict()}
                if early_stopping.early_stop:
                    print("Early stopping pre U S")
                    break
        except RuntimeError as e:
            print(epoch, "run error !!!!!")
            print(e, e.args[0])
            n_OOM += 1
    del state_probs
    del state_probs_tau
    print("now vampce score", (early_stopping.val_loss_min))
    vampnet.lobe.requires_grad_(True)
    vampnet.lobe_timelagged.requires_grad_(True)
    if best_dict:
        vampnet.vampu.load_state_dict(best_dict['vlu_dict'])
        vampnet.vamps.load_state_dict(best_dict['vls_dict'])
    # checkpoint = torch.load(modelpath)
    # vampnet.lobe.load_state_dict(checkpoint['state_dict'])
    # vampnet.lobe_timelagged = deepcopy(vampnet.lobe).to(device=device)
    # vampnet.lobe = vampnet.lobe.to(device)
    # vampnet.vampu.load_state_dict(checkpoint['vlu_dict'])
    # vampnet.vamps.load_state_dict(checkpoint['vls_dict'])
    record_result("Us step: %f, vill score %s " % (epoch, early_stopping.val_loss_min), log_pth)
    vampnet.set_optimizer_lr(0.2)  # reduce learning rate


plt.set_cmap('jet')
#print( args.trained_model, args.train , os.path.isfile(args.trained_model) )
if not args.train and os.path.isfile(args.trained_model):
    print('Loading model')
    checkpoint = torch.load(args.trained_model)
    lobe.load_state_dict(checkpoint['state_dict'])
    lobe_timelagged = deepcopy(lobe).to(device=device)
    vls.load_state_dict(checkpoint['vls_dict'])
    vls.reset_weights()
    vlu.load_state_dict(checkpoint['vlu_dict'])
    vls = vls.to(device)
    vlu = vlu.to(device)
    vlu.eval()
    vls.eval()
    lobe = lobe.to(device)
    lobe.eval()
    lobe_timelagged.eval()
    vampnet = RevVAMPNet(lobe=lobe, lobe_timelagged=lobe_timelagged, learning_rate=args.lr, device=device, optimizer='Adam',
                     score_method=args.score_method, vampu=vlu, vamps=vls)
    #vampnet = VAMPNet(lobe=lobe, lobe_timelagged=lobe_timelagged, learning_rate=args.lr, device=device)
    #train_US(train_loader=loader_train, n_epochs=args.epochs, validation_loader=loader_val,   loader_train_all=loader_train_all, model_name="us_all4")
    model = vampnet.fetch_model()

    print('Loading model done')

elif args.train:
    print("please use train.py")
    quit()
#vampnet.set_optimizer_lr(1e-1) # reduce learning rate


data_size = 0
for batch_0, batch_t in loader_train:
    data_size += batch_0.shape[0]
state_probs = np.zeros((data_size, int(args.num_classes)))
state_probs_tau = np.zeros((data_size, int(args.num_classes)))
n_iter = 0
with torch.no_grad():
    for batch_0, batch_t in loader_train:
        torch.cuda.empty_cache()
        batch_size = len(batch_0)
        # print(batch_size)
        state_probs[n_iter:n_iter + batch_size] = vampnet.transform(batch_0)
        state_probs_tau[n_iter:n_iter + batch_size] = vampnet.transform(batch_t, instantaneous=False)
        n_iter += batch_size
vampnet.set_data([state_probs,state_probs_tau])


lags = np.arange(1, 201, 2, dtype=np.int32)
its = vampnet.its(lags)
plot_its(its, lags, ylog=False, save_folder=args.save_folder)
np.save(args.save_folder+'/ITS_2.npy', np.array(its))
print("it ok")
steps = 8 + 1
tau = 40
n_classes = 4
predicted, estimated = vampnet.ck_test(steps, tau)

plot_ck_test(predicted, estimated, n_classes, steps, tau, args.save_folder)
np.savez(args.save_folder+'/ck_2.npz', list((predicted, estimated)))

print("ck ok")


quit()


