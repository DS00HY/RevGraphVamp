import argparse

import mdshare
import numpy as np
import torch
from sklearn.neighbors import BallTree
from tqdm import tqdm
from revvamp import unflatten
# from deeptime.decomposition._koopman import KoopmanChapmanKolmogorovValidator


parser = argparse.ArgumentParser()
parser.add_argument('--num-neighbors', type=int, default=5, help='number of neighbors')
parser.add_argument('--traj-folder', type=str, default=None, help='the path to the trajectory folder')
parser.add_argument('--stride', type=int, default=5, help='stride for trajectory')
# parser.add_argument('--use-backbone', action='store_true', default=False, help='Whether to use produce the data for backbone atoms')

args = parser.parse_args()
########## for loading the BBA trajectory ####################################

"""

traj_1 = ['../2JOF-0-protein/2JOF-0-protein-'+str(i).zfill(3)+'.dcd' for i in range(105)]

crys = md.load_pdb('2jof.pdb')
top = crys.topology
inds = top.select('backbone')

t1 = md.load_dcd(traj_1[0], top=top, stride=args.stride, atom_indices=inds)
coor_t1 = t1.xyz
for i in range(1,len(traj_1)):
	t1 = md.load_dcd(traj_1[i], top=top, stride=args.stride, atom_indices=inds)
	coor_t1 = np.concatenate((coor_t1, t1.xyz), axis=0)

print(coor_t1.shape)

data = list([coor_t1])
np.savez('pos_trpcage_bb.npz', data[0])
"""

if torch.cuda.is_available():
    device = torch.device('cpu')
    print('cuda is is available')
else:
    print('Using CPU')
    device = torch.device('cpu')

ala_coords_file = mdshare.fetch(
    "alanine-dipeptide-3x250ns-heavy-atom-positions.npz", working_directory="data"
)
with np.load(ala_coords_file) as fh:
    data = [fh[f"arr_{i}"].astype(np.float32) for i in range(3)]

dihedral_file = mdshare.fetch(
    "alanine-dipeptide-3x250ns-backbone-dihedrals.npz", working_directory="data"
)
#
with np.load(dihedral_file) as fh:
    dihedral = [fh[f"arr_{i}"] for i in range(3)]

# reshape the data to be in share list of [N,num_atoms,3]
data_reshaped = []
for i in range(len(data)):
    temp = data[i].reshape(data[0].shape[0], 3, 10).swapaxes(1, 2)
    data_reshaped.append(temp)
data = data_reshaped


# -----------------------------------------------------

def get_nbrs(all_coords, num_neighbors=args.num_neighbors):
    '''
    inputs: a trajectory or list of trajectories with shape [T, num_atoms, dim]
        T: number of steps
        dim: number of dimensions (3 coordinates)

    Returns:
        if all_coords is a list:
            list of trajectories of ditances and indices
        else:
            trajectory of distances and indices

        [N, num_atoms, num_neighbors]
    '''
    k_nbr = num_neighbors + 1
    if type(all_coords) == list:
        all_dists = []
        all_inds = []
        for i in range(len(all_coords)):
            dists = []
            inds = []
            tmp_coords = all_coords[i]
            for j in tqdm(range(len(tmp_coords))):
                tree = BallTree(tmp_coords[j], leaf_size=3)
                dist, ind = tree.query(tmp_coords[j], k=k_nbr)
                dists.append(dist[:, 1:])
                inds.append(ind[:, 1:])

            dists = np.array(dists)
            inds = np.array(inds)
            all_dists.append(dists)
            all_inds.append(inds)
    else:
        all_inds = []
        all_dists = []
        for i in range(len(all_coords)):
            dists = []
            inds = []
            tree = BallTree(all_coords[i], leaf_size=3)
            dist, ind = tree.query(all_coords[i], k=k_nbr)
            dists.append(dist[:, 1:])
            inds.append(ind[:, 1:])
            all_dists = np.array(dists)
            all_inds = np.array(inds)

    return all_dists, all_inds


ns = int(args.stride * 0.2)  # 0.2 ns is the timestep of trajectories


# np.savez('dists_trpcage_bb_'+str(args.num_neighbors)+'nbrs_'+ str(ns)+'ns'+'.npz', dists[0])
# np.savez('inds_trpcage_bb_'+str(args.num_neighbors)+'nbrs_'+ str(ns)+'ns'+'.npz', inds[0])
# for i in range(3):
#     np.savez('dists_trpcage_bb_' + str(args.num_neighbors) + 'nbrs_' + str(ns) + 'ns_' + str(i) + '.npz', dists[i])
#     np.savez('inds_trpcage_bb_' + str(args.num_neighbors) + 'nbrs_' + str(ns) + 'ns_' + str(i) + '.npz', inds[i])

lengths = {}
nframes = {}
lengths = np.array([a.shape[0] for a in data])
nframes = np.sum(lengths)
print(lengths)
print(nframes)
filename = "../intermediate/ala_"+str(args.num_neighbors) + 'nbrs_' + str(ns) + "ns_"
data_info = {'length': [lengths], '_nframes': nframes}
print(data_info)
np.save(filename+"datainfo.npy", data_info)

dists, inds = get_nbrs(data, args.num_neighbors)
# dists_un = [r for r in unflatten(dists, lengths)]
# inds_un = [r for r in unflatten(inds, lengths)] #into one


np.save(filename+'dist.npy', np.vstack(dists))
np.save(filename+'inds.npy', np.vstack(inds))


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
