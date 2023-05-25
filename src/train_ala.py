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
# from deeptime.decomposition._koopman import  KoopmanChapmanKolmogorovValidator
from utils_vamp import *

if torch.cuda.is_available():
	device = torch.device('cuda')
	print('cuda is available')
else:
	print('Using CPU')
	device = torch.device('cpu')

# ignore deprecation warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

args = buildParser().parse_args()


def record_result(string, file_name):
	if not os.path.exists(file_name):
		with open(file_name, 'w') as f:
			print("successfully create record file!")
			f.write(string + "\n")
	with open(file_name, 'a') as f:
		f.write(string + "\n")


log_pth = f'{args.save_folder}/training.log'

if not os.path.exists(args.save_folder):
	print('making the folder for saving checkpoints')
	os.makedirs(args.save_folder)

with open(args.save_folder + '/args.txt', 'w') as f:
	f.write(str(args))

meta_file = os.path.join(args.save_folder, 'metadata.pkl')
pickle.dump({'args': args}, open(meta_file, 'wb'))

# ------------------- data as a list of trajectories ---------------------------
file_path = args.data_path
data_info_file = file_path + "datainfo.npy"
dist_file = file_path + "dist.npy"
nbr_data_file = file_path + "inds.npy"
print(dist_file)
print(file_path)
data_info = np.load(data_info_file, allow_pickle=True).item()
traj_length = data_info['length']
print(traj_length)
#dists1, inds1 = np.load(args.dist_data)['arr_0'], np.load(args.nbr_data)['arr_0']
# dists1, inds1 = np.load(args.dist_data), np.load(args.nbr_data)
dists1, inds1 = np.load(dist_file), np.load(nbr_data_file)
print(dists1.shape)
#
# #dt = 0.25  # Trajectory timestep in ns
# #generator = DataGenerator(input_data[k], ratio=args.val_frac, dt=dt, max_frames=1000000)
#
dist_sp = [r for r in unflatten(dists1, traj_length)]
inds_sp = [r for r in unflatten(inds1, traj_length)]
data = []
print(len(dist_sp))
for i in range(len(dist_sp)):
	mydists1 = torch.from_numpy(dist_sp[i])
	myinds1 = torch.from_numpy(inds_sp[i])
	data.append(torch.cat((mydists1, myinds1), axis=-1))


dataset = TrajectoryDataset.from_trajectories(lagtime=args.tau, data=data)

n_val = int(len(dataset) * args.val_frac)
train_data, val_data = torch.utils.data.random_split(dataset, [len(dataset) - n_val, n_val])

# same traindata, val_data
print(type(train_data))

loader_train = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
loader_val = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)
# torch.save(loader_train, 'trainala.pth')
# torch.save(loader_val, 'valala.pth')
# loader_train = torch.load('trainala.pth')
# loader_val = torch.load('valala.pth')

loader_train_all = DataLoader(train_data, batch_size=len(dataset), shuffle=True)
# data is a list of trajectories [T,N,M+M]
# ---------------------------------------------------------------------------------

lobe = GraphVampNet()
# print(lobe)

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


def train(train_loader, n_epochs, validation_loader=None, loader_train_all=None):
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
		early_stopping = EarlyStopping(args.save_folder, file_name='best_pre_lobe', delta=1e-4, patience=300)
		best_dict = None
		for epoch in tqdm(range(pre_epoch)):
			try:
				for batch_0, batch_t in train_loader:
					torch.cuda.empty_cache()  # 释放显存print(batch_0.shape)
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
				if epoch % 10 == 9 :
					record_result("vamp2 step: %d, mean_sroce %f, max=%f" % (epoch, mean_score.item(), early_stopping.val_loss_min),log_pth)
			except RuntimeError as e:
				print(epoch, "run error !!!!!")
				print(e, e.args[0])
				n_OOM += 1

		vampnet.vampu.requires_grad_(True)
		vampnet.vamps.requires_grad_(True)
		vampnet.score_method = 'VAMPCE'
		if best_dict:
			vampnet.lobe.load_state_dict(best_dict)
			# vampnet.lobe_timelagged = deepcopy(lobe).to(device=device)
			vampnet.lobe_timelagged.load_state_dict(deepcopy(best_dict))
		print("vamp2 max score", (early_stopping.val_loss_min))
		record_result("vamp2 max score %f" % (early_stopping.val_loss_min), log_pth)
		# modelpath =  args.save_folder + '/best_pre_lobe.pt'
		# checkpoint = torch.load(modelpath)
		# vampnet.lobe.load_state_dict(checkpoint['state_dict'])
		# vampnet.lobe_timelagged = deepcopy(vampnet.lobe).to(device=device)
		# vampnet.lobe = vampnet.lobe.to(device)

		print("Training auxiliary network...")
		# vampnet.set_optimizer_lr(1e-1) # reduce learning rate
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
		try:
			now_train_num = 0
			for batch_0, batch_t in train_loader:
				torch.cuda.empty_cache()
				vampnet.partial_fit((batch_0.to(device), batch_t.to(device)))
				all_train_epo += 1
				now_train_num += 1

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
					best_model = {'epoch': pre_epoch, 'lobe': lobe.state_dict(), 'vlu': vlu.state_dict(),
					              'vls': vls.state_dict()}
					best_epoch = epoch
				if early_stopping.early_stop:
					print("Early stopping all network train")
					break

			if epoch % 10 == 9 and args.save_checkpoints:
				if epoch % 50 == 9:
					torch.save({
						'epoch': epoch,
						'state_dict': lobe.state_dict(),
						'vlu_dict': vlu.state_dict(),
						'vls_dict': vls.state_dict(),
					}, args.save_folder + '/logs_' + str(epoch) + '.pt')
				record_result("step: %f, mean_sroce %f , trainmean %f,  max=%f" % (vampnet._step, mean_score.item(),
				                                                          np.mean(vampnet.train_scores[
				                                                                  -now_train_num - 1:-1][0, 1]),early_stopping.val_loss_min),
				              log_pth)
		except RuntimeError as e:
			print(epoch, "run error !!!!!")
			print(e, e.args[0])
			n_OOM += 1

	if best_model:

		vampnet.lobe.load_state_dict(best_model['lobe'])
		vampnet.lobe_timelagged.load_state_dict(deepcopy(best_model['lobe']))
		# vampnet.lobe_timelagged = deepcopy(lobe).to(device=device)
		vampnet.vampu.load_state_dict(best_model['vlu'])
		vampnet.vamps.load_state_dict(best_model['vls'])
		record_result("best model score is %f " % (early_stopping.val_loss_min), log_pth)
		print("best model score is  ", early_stopping.val_loss_min)
	return vampnet.fetch_model(), all_train_epo

plt.set_cmap('jet')
# print( args.trained_model, args.train , os.path.isfile(args.trained_model) )
if not args.train and os.path.isfile(args.trained_model):
	print('Loading model')
	checkpoint = torch.load(args.trained_model)
	lobe.load_state_dict(checkpoint['state_dict'])
	lobe_timelagged = deepcopy(lobe).to(device=device)
	vls.load_state_dict(checkpoint['vls_dict'])
	vlu.load_state_dict(checkpoint['vlu_dict'])
	vls = vls.to(device)
	vlu = vlu.to(device)
	vlu.eval()
	vls.eval()
	lobe = lobe.to(device)
	lobe.eval()
	lobe_timelagged.eval()
	vampnet = RevVAMPNet(lobe=lobe, lobe_timelagged=lobe_timelagged, learning_rate=args.lr, device=device,
	                     optimizer='Adam',
	                     score_method=args.score_method, vampu=vlu, vamps=vls)
	# vampnet = VAMPNet(lobe=lobe, lobe_timelagged=lobe_timelagged, learning_rate=args.lr, device=device)
	model = vampnet.fetch_model()
	for params in vlu.parameters():
		print(params)
		print(params.data) # 获取纯数据
		print(params.shape)
	for params in vls.parameters():
		print(params)
		print(params.data) # 获取纯数据
		print(params.shape)

	print('Loading model done')

elif args.train:
	print("training model")
	model, all_train_epoch = train(train_loader=loader_train, n_epochs=args.epochs, validation_loader=loader_val,
	                               loader_train_all=loader_train_all)

	# save the training and validation scores
	with open(args.save_folder + '/train_scores.npy', 'wb') as f:
		np.save(f, vampnet.train_scores)

	with open(args.save_folder + '/validation_scores.npy', 'wb') as f:
		np.save(f, vampnet.validation_scores)

	# plotting the training and validation scores of the model

	plt.loglog(*vampnet.train_scores[-all_train_epoch:].T, label='training')
	plt.loglog(*vampnet.validation_scores.T, label='validation')
	plt.xlabel('step')
	plt.ylabel('score')
	plt.legend()
	plt.savefig(args.save_folder + '/scores.png')
	print("training model done")
# making a numpy array of data for analysis
data_np = []
for i in range(len(data)):
	data_np.append(data[i].cpu().numpy())
print("datanp",len(data_np), (data_np[0].shape))
# for the analysis part create an iterator for the whole dataset to feed in batches
whole_dataset = TrajectoryDataset.from_trajectories(lagtime=args.tau, data=data_np)
whole_dataloder = DataLoader(whole_dataset, batch_size=args.batch_size, shuffle=False)

# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
# for plotting the implied timescales
lagtimes = np.arange(1, 250, 2, dtype=np.int32)
timescales = []


#
# if True or not os.path.exists(args.save_folder+'/ITS.png'):
#
# 	for lag in tqdm(lagtimes):
# 		vamp = VAMP(lagtime=lag, observable_transform=model)
# 		whole_dataset = TrajectoryDataset.from_trajectories(lagtime=lag, data=data_np)
# 		whole_dataloder = DataLoader(whole_dataset, batch_size=1000, shuffle=False)
# 		torch.cuda.empty_cache()
# 		#print(torch.cuda.memory_summary())
# 		for batch_0, batch_t in whole_dataloder:
# 			torch.cuda.empty_cache()
# 			vamp.partial_fit((batch_0.numpy(), batch_t.numpy()))
# 	#
# 		covariances = vamp._covariance_estimator.fetch_model()
# 		ts = vamp.fit_from_covariances(covariances).fetch_model().timescales(k=5)
# 		timescales.append(ts)
#
#
#
#
# 	f, ax = plt.subplots(1, 1)
# 	ax.semilogy(lagtimes, timescales)
# 	ax.set_xlabel('lagtime')
# 	ax.set_ylabel('timescale / step')
# 	ax.fill_between(lagtimes, ax.get_ylim()[0]*np.ones(len(lagtimes)), lagtimes, alpha=0.5, color='grey');
# 	f.savefig(args.save_folder+'/ITS.png')


def chunks(data, chunk_size=5000):
	'''
    splitting the trajectory into chunks for passing into analysis part
    data: list of trajectories
    chunk_size: the size of each chunk
    '''
	if type(data) == list:

		for data_tmp in data:
			for j in range(0, len(data_tmp), chunk_size):
				print(data_tmp[j:j + chunk_size, ...].shape)
				yield data_tmp[j:j + chunk_size, ...]

	else:

		for j in range(0, len(data), chunk_size):
			yield data[j:j + chunk_size, ...]


n_classes = int(args.num_classes)

probs = []
total_emb = []
total_attn = []
for data_tmp in data_np:
	#  transforming the data into the vampnet for modeling the dynamics
	mydata = chunks(data_tmp, chunk_size=1000)
	state_probs = np.zeros((data_tmp.shape[0], n_classes))
	emb_tmp = np.zeros((data_tmp.shape[0], args.h_g))
	# emb_tmp = np.zeros((data_tmp.shape[0], args.h_a))
	attn_tmp = np.zeros((data_tmp.shape[0], args.num_atoms, args.num_neighbors))

	n_iter = 0

	for i, batch in enumerate(mydata):
		batch_size = len(batch)
		# print(batch_size)
		state_probs[n_iter:n_iter + batch_size] = model.transform(batch)
		if args.return_emb and not args.return_attn:
			emb_1 = model.lobe(torch.tensor(batch), return_emb=True, return_attn=False)
			emb_tmp[n_iter:n_iter + batch_size] = emb_1.cpu().detach().numpy()
		elif args.return_emb and args.return_attn:
			emb_1, attn_1 = model.lobe(torch.tensor(batch), return_emb=True, return_attn=True)

			emb_tmp[n_iter:n_iter + batch_size], attn_tmp[
			                                     n_iter:n_iter + batch_size] = emb_1.cpu().detach().numpy(), attn_1.cpu().detach().numpy()
		n_iter = n_iter + batch_size
	probs.append(state_probs)
	if args.return_emb:
		total_emb.append(emb_tmp)
	if args.return_attn:
		total_attn.append(attn_tmp)

# problem here
print(args.save_folder + '/transformed.npz')
np.savez(args.save_folder + '/transformed.npz', probs)
print(probs[0].shape)
print(len(probs))
if args.return_emb:
	np.savez(args.save_folder + '/embeddings.npz', total_emb)
if args.return_attn:
	np.savez(args.save_folder + '/total_attn.npz', total_attn)
np.savez(args.save_folder + '/transformed_0.npz', probs[0])
print(probs[0].shape)
print(len(probs))
if args.return_emb:
	np.savez(args.save_folder + '/embeddings_0.npz', total_emb)
if args.return_attn:
	np.savez(args.save_folder + '/total_attn_0.npz', total_attn)
print("train save ok")
# quit()

train_data, val_data = torch.utils.data.random_split(dataset, [len(dataset) - 1, 1])

# same traindata, val_data
print(type(train_data))

loader_train_1 = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
data_size = 0
for batch_0, batch_t in loader_train_1:
	data_size += batch_0.shape[0]
state_probs = np.zeros((data_size, int(args.num_classes)))
state_probs_tau = np.zeros((data_size, int(args.num_classes)))
n_iter = 0
with torch.no_grad():
	for batch_0, batch_t in loader_train_1:
		torch.cuda.empty_cache()
		batch_size = len(batch_0)
		# print(batch_size)
		state_probs[n_iter:n_iter + batch_size] = vampnet.transform(batch_0)
		state_probs_tau[n_iter:n_iter + batch_size] = vampnet.transform(batch_t, instantaneous=False)
		n_iter += batch_size
vampnet.update_auxiliary_weights([state_probs, state_probs_tau], optimize_u = False, optimize_S = False, reset_weights = False)



max_tau = 250
lags = np.arange(1, max_tau, 1)

# its = get_its(probs, lags)
# plot_its(its, lags, ylog=False, save_folder=args.save_folder)

steps = 10
tau_msm = args.tau

#predicted, estimated = vampnet.get_ck_test(probs, steps, tau_msm)
predicted, estimated = get_ck_test(probs, steps, tau_msm)

plot_ck_test(predicted, estimated, n_classes, steps, tau_msm, args.save_folder)
np.savez(args.save_folder + '/ck.npz', list((predicted, estimated)))

print("ck ok")
#its = vampnet.get_its(probs, lagtimes)
its = get_its(probs, lagtimes)

plot_its(its, lagtimes, ylog=False, save_folder=args.save_folder)
np.save(args.save_folder + '/ITS.npy', np.array(its))
print("it ok")
#
# lagtimes = np.arange(1, 250, 2, dtype=np.int32)
# timescales = []
# whole_dataset = TrajectoryDataset.from_trajectories(lagtime=args.tau, data=data_np)
# whole_dataloder = DataLoader(whole_dataset, batch_size=args.batch_size, shuffle=False)
# data_size = 0
# for batch_0, batch_t in whole_dataloder:
# 	data_size += batch_0.shape[0]
# state_probs = np.zeros((data_size, int(args.num_classes)))
# state_probs_tau = np.zeros((data_size, int(args.num_classes)))
# n_iter = 0
#
# with torch.no_grad():
# 	for batch_0, batch_t in whole_dataloder:
# 		torch.cuda.empty_cache()
# 		batch_size = len(batch_0)
# 		# print(batch_size)
# 		state_probs[n_iter:n_iter + batch_size] = vampnet.transform(batch_0)
# 		state_probs_tau[n_iter:n_iter + batch_size] = vampnet.transform(batch_t, instantaneous=False)
# 		n_iter += batch_size
# 	vampnet.update_auxiliary_weights([state_probs, state_probs_tau], optimize_S=False, optimize_u= False, reset_weights=False)
#
# steps = 10
# tau_msm = args.tau
# predicted, estimated = vampnet.ck_test(args.num_classes, steps, tau_msm)
# plot_ck_test(predicted, estimated, n_classes, steps, tau_msm, args.save_folder, 'ck_test_2.png')
# its = vampnet.its(lagtimes, args.save_folder)
# plot_its(its, lagtimes, args.save_folder)
# np.save(args.save_folder + '/ITS2.npy', np.array(timescales))
# np.savez(args.save_folder + '/ck2.npz', list((predicted, estimated)))
#
# if True or not os.path.exists(args.save_folder+'/ITS.png'):
#
# 	for lag in tqdm(lagtimes):
# 		vamp = VAMP(lagtime=lag, observable_transform=model)
# 		whole_dataset = TrajectoryDataset.from_trajectories(lagtime=lag, data=data_np)
# 		whole_dataloder = DataLoader(whole_dataset, batch_size=1000, shuffle=False)
# 		torch.cuda.empty_cache()
# 		#print(torch.cuda.memory_summary())
# 		for batch_0, batch_t in whole_dataloder:
# 			torch.cuda.empty_cache()
# 			vamp.partial_fit((batch_0.numpy(), batch_t.numpy()))
# 	#
# 		covariances = vamp._covariance_estimator.fetch_model()
# 		ts = vamp.fit_from_covariances(covariances).fetch_model().timescales(k=5)
# 		timescales.append(ts)
#
#
#
#
# 	f, ax = plt.subplots(1, 1)
# 	ax.semilogy(lagtimes, timescales)
# 	ax.set_xlabel('lagtime')
# 	ax.set_ylabel('timescale / step')
# 	ax.fill_between(lagtimes, ax.get_ylim()[0]*np.ones(len(lagtimes)), lagtimes, alpha=0.5, color='grey');
# 	f.savefig(args.save_folder+'/ITS.png')

quit()

#
# # for plotting the CK test
# vamp = VAMP(lagtime=lag, observable_transform=model)
# whole_dataset = TrajectoryDataset.from_trajectories(lagtime=200, data=data_np)
# whole_dataloder = DataLoader(whole_dataset, batch_size=10000, shuffle=False)
# for batch_0, batch_t in whole_dataloder:
# 	vamp.partial_fit((batch_0.numpy(), batch_t.numpy()))
#
# validator = chapman_kolmogorov_validator(model=vamp, mlags=10)
#
# #validator = vamp.chapman_kolmogorov_validator(mlags=5)
#
# cktest = validator.fit(data_np, n_jobs=1, progress=tqdm).fetch_model()
# n_states = args.num_classes - 1
#
# tau = cktest.lagtimes[1]
# steps = len(cktest.lagtimes)
# fig, ax = plt.subplots(n_states, n_states, sharex=True, sharey=True, constrained_layout=True)
# for i in range(n_states):
#     for j in range(n_states):
#         pred = ax[i][j].plot(cktest.lagtimes, cktest.predictions[:, i, j], color='b')
#         est = ax[i][j].plot(cktest.lagtimes, cktest.estimates[:, i, j], color='r', linestyle='--')
#         ax[i][j].set_title(str(i+1)+ '->' +str(j+1),
#                                        fontsize='small')
# ax[0][0].set_ylim((-0.1,1.1));
# ax[0][0].set_xlim((0, steps*tau));
# ax[0][0].axes.get_xaxis().set_ticks(np.round(np.linspace(0, steps*tau, 3)));
# fig.legend([pred[0], est[0]], ["Predictions", "Estimates"], 'lower center', ncol=2,
#            bbox_to_anchor=(0.5, -0.1));
#
# fig.savefig(args.save_folder+'/cktest.png')
