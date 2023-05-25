import argparse
import mdtraj as md
from glob import glob
import mdshare
import numpy as np
from typing import List
import torch
from sklearn.neighbors import BallTree
from revvamp import unflatten
from tqdm import tqdm
import os
#from deeptime.decomposition._koopman import KoopmanChapmanKolmogorovValidator
import pyemma as pe

parser = argparse.ArgumentParser()
parser.add_argument('--num-neighbors', type=int, default=5, help='number of neighbors')
parser.add_argument('--traj-folder', type=str, default=None, help='the path to the trajectory folder')
parser.add_argument('--stride', type=int, default=5, help='stride for trajectory')
#parser.add_argument('--use-backbone', action='store_true', default=False, help='Whether to use produce the data for backbone atoms')

args = parser.parse_args()
########## for loading the BBA trajectory ####################################


def get_nbrs(all_coords, pair_list, num_atoms=42, num_neighbors=args.num_neighbors):
    '''
    inputs: a trajectory or list of trajectories with shape [T, n_redius_pair]
        T: number of steps
        n_redius_pair: number of dimensions  input ($\frac{N(N-1)}{2}$ values, 780
        pair_list: n_redius_pair

    Returns:
        if all_coords is a list:
            list of trajectories of ditances and indices
        else:
            trajectory of distances and indices

        [N, num_atoms, num_neighbors]
    '''
    print(type(all_coords))
    if type(all_coords) == list:
        all_dists = []
        all_inds = []
        for i in range(len(all_coords)):
            dists = []
            inds = []
            tmp_coords = all_coords[i]
            for j in tqdm(range(len(tmp_coords))):
                mut_dist = np.ones(shape=(num_atoms,num_atoms)) *100.0
                for idx , d in enumerate(tmp_coords[j]):
                    res_i, res_j = pair_list[idx]
                    mut_dist[res_i-1][res_j-1] = d
                    mut_dist[res_j-1][res_i-1] = d
                dist = []
                ind = []
                for dd in mut_dist:
                    states_order = np.argsort(dd)
                    res = list(states_order[:num_neighbors])
                    dist.append(list(np.array(dd)[res]))
                    ind.append(res)
                dists.append(dist)
                inds.append(ind)
            dists = np.array(dists)
            inds = np.array(inds)
            all_dists.append(dists)
            all_inds.append(inds)
    else:
        all_dists = []
        all_inds = []

        for j in tqdm(range(len(all_coords))):
            mut_dist = np.ones(shape=(num_atoms, num_atoms)) * 300.0
            for idx, d in enumerate(all_coords[j]):
                res_i, res_j = pair_list[idx]
                mut_dist[res_i][res_j] = d
                mut_dist[res_j][res_i] = d
            dist = []
            ind = []
            for dd in mut_dist:
                states_order = np.argsort(dd)
                res = list(states_order[:num_neighbors])
                dist.append(list(np.array(dd)[res]))
                ind.append(res)
            all_dists.append(dist)
            all_inds.append(ind)
        all_dists = np.array(all_dists)
        all_inds = np.array(all_inds)
    return all_dists, all_inds

sim_names = ("red", "ox")
top, trajs = {}, {}
trajs = {k: sorted(glob("D:/code/ab42-kinetic-ensemble-master/trajectories/trajectories/{0}/r?/traj*.xtc".format(k))) for k in sim_names}
top = {k: "D:/code/ab42-kinetic-ensemble-master/trajectories/trajectories/{0}/topol.gro".format(k) for k in sim_names}
KBT = 2.311420  # 278 K
nres = 42
traj_rounds = {
    "red": [1024, 1023, 1024, 1024, 1024],
    "ox": [1024, 1024, 1023],
}

residue_name = {}
pair_list = {}
inpcon = {}
for k in sim_names:
    residue = {}
    pair = []
    feat = pe.coordinates.featurizer(top[k])
    feat.add_residue_mindist()
    for key in feat.describe():
        name = key.split(' ')
        ri, rj = name[2], name[4]
        i, j = int(ri[3:]), int(rj[3:])
        residue[i] = ri
        residue[j] = rj
        pair.append((i,j))
    residue_name[k] = residue
    pair_list[k] = pair
    inpcon[k] = pe.coordinates.source(trajs[k], feat)
    print(residue.values())
    print(pair)

# %%

lengths, nframes = {}, {}
for i, k in enumerate(sim_names):
    # Switch for full data:
    lengths[k] = [inpcon[k].trajectory_lengths()]
    #     lengths[k] = sort_lengths(inpcon[k].trajectory_lengths(), traj_rounds[k])
    nframes[k] = inpcon[k].trajectory_lengths().sum()



print("\t\t" + "\t\t".join(sim_names))
print("\n".join((
    "Trajs: \t\t" + "\t\t".join("{0}".format(len(trajs[k])) for k in sim_names),
    "Frames: \t" + "\t\t".join("{0}".format(nframes[k]) for k in sim_names),
    "Time: \t\t" + "\t".join("{0:5.3f} µs".format(inpcon[k].trajectory_lengths().sum() * 0.00025)
                             for k in sim_names)
)))

# %% md

for k in sim_names:
    filename = "../intermediate/mindist-780-{0}.npy".format(k)
    if not os.path.exists(filename):
        print("No mindist file for {0} ensemble, calculating from scratch...".format(k))
        con = np.vstack(inpcon[k].get_output())
        np.save(filename, con)

# # %%
#
input_flat, input_data = {}, {}
data = {}
for k in sim_names:
    raw = np.load("../intermediate/mindist-780-{0}.npy".format(k))
    data[k] = [r for r in unflatten(raw, lengths[k])]
    raw_mean, raw_std = raw.mean(axis=0), raw.std(axis=0)
    input_flat[k] = (raw - raw_mean) / raw_std
    input_data[k] = [(r - raw_mean) / raw_std for r in unflatten(raw, lengths[k])]

for k in sim_names:
    filename = "../intermediate/dist-780-{0}.npy".format(k)
    if not os.path.exists(filename):
        print("No mindist file for {0} ensemble, calculating from scratch...".format(k))
        con = np.vstack(inpcon[k].get_output())
        np.save(filename, con)



ns = int(args.stride*0.2) # 0.2 ns is the timestep of trajectories
lengths = {}
nframes = {}
for k in sim_names:
    filename = "../intermediate/"+k +"_"+str(args.num_neighbors) + 'nbrs_' + str(ns) + "ns_"
    dists, inds = get_nbrs(data[k], pair_list[k], 42, 10)

    lengths[k] = [inpcon[k].trajectory_lengths()]
    nframes[k] = inpcon[k].trajectory_lengths().sum()
    data_info = {'length': lengths[k], '_nframes': nframes[k]}
    print(data_info)
    np.save(filename+"datainfo.npy", data_info)
    np.save(filename+'dist.npy', np.vstack(dists))
    np.save(filename+'inds.npy', np.vstack(inds))
    # for i in range(len(dists)):
    #     np.savez(k+'dists_rap_'+str(args.num_neighbors)+'nbrs_'+ str(ns)+'ns_'+str(i)+'.npz', dists[i])
    #     np.savez(k+'inds_rap'+str(args.num_neighbors)+'nbrs_'+ str(ns)+'ns_'+str(i)+'.npz', inds[i])

def chunks(data, chunk_size=5000):
    '''
    splitting the trajectory into chunks for passing into analysis part
    data: list of trajectories
    chunk_size: the size of each chunk
    '''
    if type(data) == list:

        for data_tmp in data:
            for j in range(0, len(data_tmp),chunk_size):
                print(data_tmp[j:j+chunk_size,...].shape)
                yield data_tmp[j:j+chunk_size,...]

    else:

        for j in range(0, len(data), chunk_size):
            yield data[j:j+chunk_size,...]
