from deeptime.util.data import TrajectoryDataset
from deeptime.decomposition.deep import *
from tqdm import tqdm
from deeptime.util.torch import disable_TF32, multi_dot
from deeptime.decomposition import VAMP
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from typing import Optional, Union, Callable, Tuple
from args import buildParser
import h5py
from typing import Tuple, Sequence, List, Union, Generator, Callable, Any, Dict, TypeVar, Set
from pathlib import Path

import itertools
from torch.utils.data import DataLoader
import os

args = buildParser().parse_args()

if torch.cuda.is_available():
	device = torch.device('cuda')
	print('cuda is is available')
else:
	print('Using CPU')
	device = torch.device('cpu')

FRAMES, DIMENSIONS, FIRST, LAST, TIME, TIME_PLUS_LAG = 0, 1, 0, -1, 0, 1
LAG_EPOCH = 1000
T = TypeVar("T")
MaybeListType = Union[List[T], T]
NNDataType = Tuple[List[np.ndarray], np.ndarray]
MaybePathType = Union[Path, str]

def handle_path(path: MaybePathType, non_existent: bool=False) -> Path:
    """
    Check path validity and return `Path` object.

    Parameters
    ----------
    path
        Filepath to be checked.
    non_existent
        If false, will raise an error if the path does not exist.

    Returns
    -------
    path
        The converted and existing path.

    """
    if not isinstance(path, Path):
        try:
            path = Path(path)
        except Exception as err:
            message = "Couldn't read path {0}! Original message: {1}"
            raise ValueError(message.format(path, err))
    if not path.exists() and not non_existent:
        raise IOError("File {0} does not exist!".format(path))
    if not path.parent.exists():
        path.parent.mkdir()
    return path

def unflatten(source: np.ndarray, lengths: List[int]) -> List[np.ndarray]:
    """
    Takes an array and returns a list of arrays.

    Parameters
    ----------
    source
        Array to be unflattened.
    lengths
        List of integers giving the length of each subarray.
        Must sum to the length of source.

    Returns
    -------
    unflat
        List of arrays.

    """
    conv = []
    lp = 0
    for arr in lengths:
        arrconv = []
        for le in arr:
            arrconv.append(source[lp:le + lp])
            lp += le
        conv.append(arrconv)
    ccs = list(itertools.chain(*conv))
    return ccs

def make_list(item: MaybeListType[T], cls=list) -> List[T]:
    """
    Turn an object into a list, if it isn't already.

    Parameters
    ----------
    item
        Item to contain in a list

    Returns
    -------
    list
        List with item as only element

    """
    if not isinstance(item, list):
        item = [item]
    return cls(item)


VALIDS = {int, float, str, list}


def _get_serializable_attributes(obj: object) -> Dict[str, Any]:
    """
    Finds all object attributes that are serializable with HDF5.

    Parameters
    ----------
    obj
        Object to serialize

    Returns
    -------
    attributes
        All serializable public attributes

    """
    return {k: v for k, v in obj.__dict__.items()
            if any(isinstance(v, valid) for valid in VALIDS)
            and not k.startswith("_")}

class DataGenerator:
    def __init__(self, data: MaybeListType[np.ndarray],
                 ratio: float = 0.9, dt: float = 1.0, max_frames: int = None):
        """
        DataGenerator - Produces data for training a Koopman model.

        Parameters
        ----------
        data
            Input data as (a list of) ndarrays with
            frames as rows and features as columns
        ratio
            Train / validation split ratio
        dt
            Timestep of the underlying data
        max_frames
            The maximum number of frames to use

        """
        self._data = make_list(data)
        self.ratio = ratio
        self.dt = dt
        self.max_frames = max_frames or self.n_points

        # Generate lag = 0 indices, we will use these for different
        # lag times later. That way we can retrain with essentially
        # the same data for different lag times.
        self.regenerate_indices()

    @property
    def data(self) -> List[np.ndarray]:
        return self._data

    @property
    def n_dims(self) -> int:
        """Number of dimensions in the input data."""
        return self.data[FIRST].shape[DIMENSIONS]

    @property
    def n_points(self) -> int:
        """Number of frames in the input data."""
        return sum(self.traj_lengths)

    @property
    def n_traj(self) -> int:
        """Number of trajectories in the input data."""
        return len(self.data)

    @property
    def traj_lengths(self) -> int:
        """Length of all trajectories in the input data."""
        return [len(t) for t in self.data]

    @property
    def data_flat(self) -> np.ndarray:
        """The flattened input data."""
        return np.vstack(self.data)

    @classmethod
    def from_state(cls, data: MaybeListType[np.ndarray],
                   filename: MaybePathType) -> "DataGenerator":
        """
        Creates a DataGenerator object from previously saved index data.

        Parameters
        ----------
        data
            Input data as (a list of) ndarrays with
            frames as rows and features as columns
        filename
            File to load the indices from.

        """
        gen = cls(data)
        gen.load(filename)

        # Check for data consistency
        assert gen.n_traj == len(data), "Inconsistent data lengths!"
        assert all(len(gen._indices[i]) == gen.traj_lengths[i]
                   for i in range(gen.n_traj)), "Inconsistent trajectory lengths!"
        return gen

    def regenerate_indices(self):
        """Regenerate random indices."""
        # We use a dict here because we might otherwise desync
        # our indices and trajectories when generating the
        # train and test data. This way we're sure we're
        # accessing the correct indices.
        self._indices = {}
        for i, traj in enumerate(self.data):
            inds = np.arange(traj.shape[FRAMES])
            np.random.shuffle(inds)
            self._indices[i] = inds

        # We will also shuffle the whole dataset to avoid
        # preferentially sampling late round trajectories.
        # These are more indices than we will need in practice,
        # because the trajectories are shortened through the
        # lag time. We will just cut out the extra ones later.
        self._full_indices = np.random.choice(
            np.arange(self.max_frames), size=self.max_frames, replace=False)

    def truncate_indices(self, index: int):
        """
        Truncate the indices up to a maximum entry.
        Useful for generating convergence data.

        Parameters
        ----------
        index
            Maximum index to use

        """
        for idx in self._full_indices[self._full_indices > index]:
            del self._indices[idx]

        self._full_indices = self._full_indices[self._full_indices < index]

    def save(self, filename: MaybePathType):
        """
        Save the generator state in the form of indices.

        Parameters
        ----------
        filename
            File to save the indices to.

        """
        with h5py.File(handle_path(filename, non_existent=True), "w") as write:
            # Save the individual trajectory indices
            inds = write.create_group("indices")
            for k, v in self._indices.items():
                inds[str(k)] = v

            # Save the indices on a trajectory level
            dset = write.create_dataset("full_indices", data=self._full_indices)
            dset.attrs.update(_get_serializable_attributes(self))

    def load(self, filename: MaybePathType):
        """
        Load the generator state from indices.

        Parameters
        ----------
        filename
            File to load the indices from.

        """
        with h5py.File(handle_path(filename), "r") as read:
            # Object state (ratio etc...)
            self.__dict__.update(read["full_indices"].attrs)
            self._full_indices = read["full_indices"][:]

            # All indices
            self._indices = {int(k): v[:] for k, v in read["indices"].items()}

    def _generate_indices(self, lag: int) -> Dict[int, np.ndarray]:
        """
        Generates indices corresponding to a particular lag time.

        Parameters
        ----------
        lag
            The lag time for data preparation

        Returns
        -------
        indices
            Dictionary of trajectory indices with selected frames

        """
        indices = {}
        for k, inds in self._indices.items():
            max_points = inds.shape[FRAMES] - lag

            # Lag time longer than our trajectory
            if max_points <= 0:
                continue

            indices[k] = inds[inds < max_points]
        return indices

    def __call__(self, n: int, lag: int):
        """
        Creates the data for training the neural network.

        Parameters
        ----------
        n
            The size of the output
        lag
            The lag time in steps to be used

        Returns
        -------
        data
            DataSet of training and test data

        """
        xt_shuf = []
        xttau_shuf = []
        indices = self._generate_indices(lag)

        selected_indices = []
        total_points = 0
        for i, traj in enumerate(self.data):
            n_points = traj.shape[FRAMES]

            # We'll just skip super short trajectories for now
            if n_points <= lag:
                continue

            xt = traj[:n_points - lag]
            xttau = traj[lag:]
            xt_shuf.append(xt[indices[i]])
            xttau_shuf.append(xttau[indices[i]])

            # Collect all selected indices for debugging
            selected_indices.append(indices[i] + total_points)
            total_points += n_points

        self._selected_indices = np.concatenate(selected_indices)

        xt = np.vstack(xt_shuf).astype(np.float32)
        xttau = np.vstack(xttau_shuf).astype(np.float32)

        eff_len = min(xt.shape[FRAMES], self.max_frames)
        train_len = int(np.floor(eff_len * self.ratio))

        # Reshuffle to remove trajectory level bias
        inds = self._full_indices[self._full_indices < eff_len]
        xt, xttau = xt[inds], xttau[inds]
        self._selected_indices = self._selected_indices[inds][:train_len]

        return [xt[:train_len], xttau[:train_len]], [xt[train_len:eff_len], xttau[train_len:eff_len]]



class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, save_path, file_name='best_network', patience=15, verbose=False, delta=1e-6):
        """
        Args:
            save_path : 模型保存文件夹
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.save_path = save_path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.file_name = file_name
        self.is_best = False

    def __call__(self, val_loss, model):

        #score = -val_loss
        score = val_loss

        if np.isnan(score):
            self.is_best = False
            print(f'now is Nan !! !EarlyStopping counter: {self.counter} out of {self.patience}')
            self.early_stop = True
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.is_best = True
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                self.early_stop = True
            self.is_best = False
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
            self.is_best = True


    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        path = os.path.join(self.save_path, "{}.pt".format(self.file_name))
        if isinstance(model, dict):
            torch.save(model, path)
        else:
            torch.save(model.state_dict(), path)	# save beat model
        self.val_loss_min = val_loss

class VAMPU(nn.Module):
    def __init__(self, units, activation, **kwargs):
        self.M = units
        self.activation = activation
        super().__init__(**kwargs)
        #params = torch.nn.ParameterDict()
        #params.update({"u_var": nn.Parameter(Variable(torch.ones(2, 3)))})

        self._u_kernel = torch.nn.Parameter((1. / self.M) * torch.ones((self.M,)), requires_grad=True)# u_var

    @ property
    def u_kernel(self) -> torch.nn.Parameter:
        r"""
        :type:
        """
        return self._u_kernel

    def compute_output_shape(self, input_shape):
        return [self.M] * 2 + [(self.M, self.M)] * 4 + [self.M]

    def _tile(self, x, n_batch):
        x_exp = torch.unsqueeze(x, axis=0) #
        shape = x.shape
        return torch.tile(x_exp, [n_batch, *([1] * len(shape))])

    def forward(self, x):
        chi_t, chi_tau = x
        n_batch = chi_t.shape[0]
        norm = 1. / n_batch
        chi_tau_t = chi_tau.t()
        corr_tau = norm * torch.matmul(chi_tau_t, chi_tau)
        chi_mean = torch.mean(chi_tau, axis=0, keepdims=True) #reduce_mean
        ac_u_kernel = self.activation(self._u_kernel).to(device)
        kernel_u = torch.unsqueeze(ac_u_kernel, axis=0)

        #tmp_p = torch.sum(chi_mean * kernel_u, 1, keepdim=True)
        u = kernel_u / torch.sum(chi_mean * kernel_u, 1, keepdims=True) ### ！！！！！
        u_t = u.t()
        v = torch.matmul(corr_tau, u_t)
        mu = norm * torch.matmul(chi_tau, u_t)#, transpose_b=True)
        cmu_t = (chi_tau * mu).t()
        sigma = torch.matmul(cmu_t, chi_tau)#, transpose_a=True)
        gamma = chi_tau * torch.matmul(chi_tau, u_t)#, transpose_b=True)
        gamma_t = gamma.t()

        chi_t_t = chi_t.t()
        #C00, C11, C01 = covariances(chi_t, chi_tau, remove_mean=True)
        C00 = norm * torch.matmul(chi_t_t, chi_t)#, transpose_a=True)
        C11 = norm * torch.matmul(gamma_t, gamma)#, transpose_a=True)
        C01 = norm * torch.matmul(chi_t_t, gamma)#, transpose_a=True)
        # print(corr_tau.shape)
        # print(chi_mean.shape)
        # print(kernel_u.shape)
        # print(gamma.shape)
        # print(tmp_p.shape)

        return [
                   self._tile(var, n_batch) for var in (u, v, C00, C11, C01, sigma)
               ] + [mu]


# In[9]:

def loss_vampe(y_true, y_pred):
    return torch.trace(y_pred[0])


class VAMPS(nn.Module):
    def __init__(self, units, activation, order=20, renorm=False, **kwargs):
        self.M = units
        self.activation = activation
        self.renorm = renorm
        self.order = order

        super().__init__(**kwargs)
        # params = torch.nn.ParameterDict()
        # params.update({"S_var": nn.Parameter(Variable(torch.ones(2, 3)))})

        self._s_kernel = torch.nn.Parameter(0.1 * torch.ones((self.M, self.M)), requires_grad=True)  # S_var
        self._init_weight = None

    @property
    def s_kernel(self) -> torch.nn.Parameter:
        r"""
        :type:
        """
        return self._s_kernel

    def reset_weights(self):
        if self._init_weight is None:
            self._init_weight = self._s_kernel
        else:
            self._s_kernel = self._init_weight


    def compute_output_shape(self, input_shape):
        return [(self.M, self.M)] * 2 + [self.M] + [(self.M, self.M)]

    def forward(self, x):
        if len(x) == 5:
            v, C00, C11, C01, sigma = x
        else:
            chi_t, chi_tau, u, v, C00, C11, C01, sigma = x
            u = u[0]

        n_batch = v.shape[0]
        norm = 1. / n_batch
        C00, C11, C01 = C00[0], C11[0], C01[0]
        sigma, v = sigma[0], v[0]

        kernel_w = self.activation(self._s_kernel).to(device)
        kernel_w_t = kernel_w.t()
        w1 = kernel_w + kernel_w_t
        w_norm = w1 @ v

        # Numerical problems with using a high p-norm
        if self.renorm:
            quasi_inf_norm = lambda x: torch.max(torch.abs(x)) #
            w1 = w1 / quasi_inf_norm(w_norm)
            w_norm = w1 @ v

        w2 = (1 - torch.squeeze(w_norm)) / torch.squeeze(v)
        S = w1 + torch.diag(w2) #

        S_t = S.t()
        u_t = u.t()
        chi_tau_t = chi_tau.t()

        if len(x) == 8:
            q = (norm * torch.matmul(S, chi_tau_t).t()
                 * torch.matmul(chi_tau, u_t))
            probs = torch.sum(chi_t * q, axis=1)

        K = S @ sigma
        vamp_e = S_t @ C00 @ S @ C11 - 2 * S_t @ C01
        vamp_e_tile = torch.tile(torch.unsqueeze(vamp_e, axis=0), [n_batch, 1, 1]) # tf.expand_dims
        K_tile = torch.tile(torch.unsqueeze(K, axis=0), [n_batch, 1, 1])
        S_tile = torch.tile(torch.unsqueeze(S, axis=0), [n_batch, 1, 1])
        if len(x) == 5:
            return [vamp_e_tile, K_tile, torch.zeros((n_batch, self.M)), S_tile]
        else:
            return [vamp_e_tile, K_tile, probs, S_tile]


# In[10]:
valid_score_methods = ('VAMP1', 'VAMP2', 'VAMPE', 'VAMPCE')

def vamp_score(data: torch.Tensor, data_lagged: torch.Tensor, method='VAMP2', epsilon: float = 1e-6, mode='trunc'):
    r"""Computes the VAMP score based on data and corresponding time-shifted data.

    Parameters
    ----------
    data : torch.Tensor
        (N, d)-dimensional torch tensor
    data_lagged : torch.Tensor
        (N, k)-dimensional torch tensor
    method : str, default='VAMP2'
        The scoring method. See :meth:`score <deeptime.decomposition.CovarianceKoopmanModel.score>` for details.
    epsilon : float, default=1e-6
        Cutoff parameter for small eigenvalues, alternatively regularization parameter.
    mode : str, default='trunc'
        Regularization mode for Hermetian inverse. See :meth:`sym_inverse`.

    Returns
    -------
    score : torch.Tensor
        The score. It contains a contribution of :math:`+1` for the constant singular function since the
        internally estimated Koopman operator is defined on a decorrelated basis set.
    """
    assert method in valid_score_methods, f"Invalid method '{method}', supported are {valid_score_methods}"
    assert data.shape == data_lagged.shape, f"Data and data_lagged must be of same shape but were {data.shape} " \
                                            f"and {data_lagged.shape}."
    out = None
    if method == 'VAMP1':
        koopman = koopman_matrix(data, data_lagged, epsilon=epsilon, mode=mode)
        out = torch.norm(koopman, p='nuc')
    elif method == 'VAMP2':
        koopman = koopman_matrix(data, data_lagged, epsilon=epsilon, mode=mode)
        out = torch.pow(torch.norm(koopman, p='fro'), 2)
    elif method == 'VAMPE':
        c00, c0t, ctt = covariances(data, data_lagged, remove_mean=True)
        c00_sqrt_inv = sym_inverse(c00, epsilon=epsilon, return_sqrt=True, mode=mode)
        ctt_sqrt_inv = sym_inverse(ctt, epsilon=epsilon, return_sqrt=True, mode=mode)
        koopman = multi_dot([c00_sqrt_inv, c0t, ctt_sqrt_inv]).t()

        u, s, v = torch.svd(koopman)
        mask = s > epsilon

        u = torch.mm(c00_sqrt_inv, u[:, mask])
        v = torch.mm(ctt_sqrt_inv, v[:, mask])
        s = s[mask]

        u_t = u.t()
        v_t = v.t()
        s = torch.diag(s)
        out = torch.trace(
            2. * multi_dot([s, u_t, c0t, v]) - multi_dot([s, u_t, c00, u, s, v_t, ctt, v])
        )
    elif method == 'VAMPCE':
        #out = torch.trace(data)
        out = torch.trace(data[0])
        assert out is not None
        return out * -1.0
    assert out is not None
    return 1 + out


def vampnet_loss(data: torch.Tensor, data_lagged: torch.Tensor, method='VAMP2', epsilon: float = 1e-6,
                 mode: str = 'trunc'):
    r"""Loss function that can be used to train VAMPNets. It evaluates as :math:`-\mathrm{score}`. The score
    is implemented in :meth:`score`."""
    return -1. * vamp_score(data, data_lagged, method=method, epsilon=epsilon, mode=mode)


def matrix_inverse(mat, epsilon=1e-10):
    """
    Calculates the inverse of a square matrix.

    Parameters
    ----------
    mat
        Square real matrix

    Returns
    -------
    inv
        Inverse of the matrix

    """
    mat_cpu = mat.detach().to('cpu')
    eigva, eigveca = np.linalg.eigh(mat_cpu.numpy())
    inc = eigva > epsilon
    eigv, eigvec = eigva[inc], eigveca[:, inc]
    return eigvec @ np.diag(1. / eigv) @ eigvec.T

def covariances_E(chil, chir):
    """
    Calculates (lagged) covariances.

    Parameters
    ----------
    data
        Data at time t and t + tau

    Returns
    -------
    C0inv
        Inverse covariance
    Ctau
        Lagged covariance

    """
    norm = 1. / chil.shape[0]
    C0, Ctau = norm * chil.T @ chil, norm * chil.T @ chir
    C0inv = matrix_inverse(C0)
    return C0inv, Ctau


def _compute_pi(K):
    """
    Calculates the stationary distribution of a transition matrix.

    Parameters
    ----------
    K
        Transition matrix

    Returns
    -------
    pi
        Normalized stationary distribution

    """
    eigv, eigvec = np.linalg.eig(K.T)
    pi_v = eigvec[:, ((eigv - 1) ** 2).argmin()]
    return pi_v / pi_v.sum(keepdims=True)

class RevVAMPNet(VAMPNet):

    def __init__(self, lobe: nn.Module, lobe_timelagged: Optional[nn.Module] = None, vampu = None, vamps = None,
                 device=None, optimizer: Union[str, Callable] = 'Adam', learning_rate: float = 5e-4,
                 score_method: str = 'VAMP2', score_mode: str = 'regularize', epsilon: float = 1e-6,
                 dtype=np.float32, n_output=args.num_classes):

        super().__init__(lobe, lobe_timelagged, device, optimizer, learning_rate,
                         score_method, score_mode, epsilon, dtype)
        self.n_output = n_output
        self._vampu = vampu
        self._vamps = vamps
        self._k_cache = {}
        self.network_lag = args.tau
        self._lag = args.tau
        self._K = None
        self.data = None
        if score_method == 'VAMPCE':
            assert vampu != None and vamps != None, f"vampu and vamps module must be defined "
            self.setup_optimizer(optimizer, list(self.lobe.parameters()) + list(self.lobe_timelagged.parameters()) +
                                 list(self._vampu.parameters()) + list(self._vamps.parameters()))

    @property
    def K(self) -> np.ndarray:
        """The estimated Koopman operator."""
        if self._K is None or self._reestimated:
            self._K = np.ones((1,1))

        return self._K

    @property
    def vampu(self) -> nn.Module:
        r""" The vampu module

        Returns
        -------
        lobe : nn.Module
        """
        return self._vampu

    @property
    def vamps(self) -> nn.Module:
        r""" The vamps lobe.

        Returns
        -------
        lobe : nn.Module
        """
        return self._vamps


    def score_method(self, value: str):
        assert value in valid_score_methods, f"Tried setting an unsupported scoring method '{value}', " \
                                             f"available are {valid_score_methods}."
        self._score_method = value


    def partial_fit(self, data, train_score_callback: Callable[[int, torch.Tensor], None] = None):
    #def rev_fit(self, data, train_score_callback: Callable[[int, torch.Tensor], None] = None):
        """
        Parameters
        ----------
        data : tuple or list of length 2, containing instantaneous and timelagged data
            The data to train the lobe(s) on.
        train_score_callback : callable, optional, default=None
            An optional callback function which is evaluated after partial fit, containing the current step
            of the training (only meaningful during a :meth:`fit`) and the current score as torch Tensor.

        Returns
        -------
        self : VAMPNet
            Reference to self.
        """
        if self.dtype == np.float32:
            self._lobe = self._lobe.float()
            self._lobe_timelagged = self._lobe_timelagged.float()
        elif self.dtype == np.float64:
            self._lobe = self._lobe.double()
            self._lobe_timelagged = self._lobe_timelagged.double()

        self.lobe.train()
        self.lobe_timelagged.train()

        assert isinstance(data, (list, tuple)) and len(data) == 2, \
            "Data must be a list or tuple of batches belonging to instantaneous " \
            "and respective time-lagged data."

        batch_0, batch_t = data[0], data[1]

        if isinstance(data[0], np.ndarray):
            batch_0 = torch.from_numpy(data[0].astype(self.dtype)).to(device=self.device)
        if isinstance(data[1], np.ndarray):
            batch_t = torch.from_numpy(data[1].astype(self.dtype)).to(device=self.device)

        self.optimizer.zero_grad()
        x_0 = self.lobe(batch_0)
        x_t = self.lobe_timelagged(batch_t)
        if self.score_method == 'VAMPCE':
            self._vampu.train()
            self._vamps.train()
            self._vampu.u_kernel.retain_grad()
            self._vamps.s_kernel.retain_grad()
            (u_out, v_out, C00_out, C11_out,
             C01_out, sigma_out, mu_out) = self._vampu([x_0, x_t])
            Ve_out, K_out, p_out, S_out = self._vamps([
                x_0, x_t, u_out, v_out, C00_out,
                C11_out, C01_out, sigma_out])
            # (u_out, v_out, C00_out, C11_out,
            #    C01_out, sigma_out) = self._vampu([x_0, x_t])
            # Ve_out = self._vamps([
            #     x_0, x_t, u_out, v_out, C00_out,
            #     C11_out, C01_out, sigma_out])
            self._K = K_out[0]
            loss_value = vampnet_loss(Ve_out, Ve_out, method=self.score_method, epsilon=self.epsilon, mode=self.score_mode)
        else:
            loss_value = vampnet_loss(x_0, x_t, method=self.score_method, epsilon=self.epsilon, mode=self.score_mode)

        #torch.autograd.set_detect_anomaly(True)
        # 反向传播时检测是否有异常值，定位code
        #print(Ve_out[0])
        #print(loss_value.item())
        #with torch.autograd.detect_anomaly():
        #    loss_value.backward()

        loss_value.backward()
        self.optimizer.step()

        if train_score_callback is not None:
            lval_detached = loss_value.detach()
            train_score_callback(self._step, -lval_detached)
        self._train_scores.append((self._step, (-loss_value).item()))
        self._step += 1

        return self


    def validate(self, validation_data: Tuple[torch.Tensor]) -> torch.Tensor:
        r""" Evaluates the currently set lobe(s) on validation data and returns the value of the configured score.

        Parameters
        ----------
        validation_data : Tuple of torch Tensor containing instantaneous and timelagged data
            The validation data.

        Returns
        -------
        score : torch.Tensor
            The value of the score.
        """
        with disable_TF32():
            self.lobe.eval()
            self.lobe_timelagged.eval()
            if self.vamps != None:
                self.vamps.eval()
                self.vampu.eval()

            with torch.no_grad():
                val = self.lobe(validation_data[0])
                val_t = self.lobe_timelagged(validation_data[1])
                if self.score_method == 'VAMPCE':
                    (u_out, v_out, C00_out, C11_out,
                     C01_out, sigma_out, mu_out) = self._vampu([val, val_t])
                    Ve_out, K_out, p_out, S_out = self._vamps([
                        val, val_t, u_out, v_out, C00_out,
                        C11_out, C01_out, sigma_out])
                    # (u_out, v_out, C00_out, C11_out,
                    #  C01_out, sigma_out) = self._vampu([x_0, x_t])
                    # Ve_out = self._vamps([
                    #     x_0, x_t, u_out, v_out, C00_out,
                    #     C11_out, C01_out, sigma_out])
                    score_value = vamp_score(Ve_out, Ve_out, method=self.score_method, mode=self.score_mode, epsilon=self.epsilon)
                else:
                    score_value = vamp_score(val, val_t, method=self.score_method, mode=self.score_mode, epsilon=self.epsilon)

                return score_value

    def update_auxiliary_weights(self, data, optimize_u: bool = True, optimize_S: bool = False,
                                  reset_weights: bool = True):
        """
        Update the weights for the auxiliary model and return new output

        Parameters
        ----------
        data
            chi [chi, chi_t]
        optimize_u
            Whether to optimize the u vector
        optimize_S
            Whether to optimize the S matrix
        reset_weights
            Whether to reset the weights for the vanilla VAMPNet model

        Returns
        -------
        chi
            New training and validation assignments

        """
        # if reset_weights:
        #     self._models["chi"].weights = self._chi_weights

        # Project training data


        batch_0, batch_t = data[0], data[1]
        chi_0 = torch.Tensor(batch_0).to(device)
        chi_t = torch.Tensor(batch_t).to(device)
        # chi_0 = self.lobe(batch_0)
        # chi_t = self.lobe_timelagged(batch_t)
       # np.save('chi_0.npy',chi_0.to('cpu'))
        #np.save('chi_t.npy', chi_t.to('cpu'))
        # Set weights for u vector
        C0inv, Ctau = covariances_E(chi_0, chi_t)


        (u_outd, v_outd, C00_outd, C11_outd,
         C01_outd, sigma_outd, mu_outd) = self._vampu([chi_0, chi_t])
        Ve_out, K_out, p_out, S_out = self._vamps([
            chi_0, chi_t, u_outd, v_outd, C00_outd,
            C11_outd, C01_outd, sigma_outd])
        #print(Ve_out.shape)

        K = torch.Tensor(C0inv) @ Ctau.to('cpu')
        self._K = K_out[0]
        #np.save('u_init.npy',self.vampu.u_kernel.detach().to('cpu'))
        #np.save('s_init.npy', self.vamps.s_kernel.detach().to('cpu'))
        if optimize_u:
            pi = _compute_pi(K)
            u_kernel = np.log(np.abs(C0inv @ pi))
            #print(self.vampu.u_kernel)
            #np.save('u_kernel.npy', u_kernel)
            for param in self.vampu.parameters():
                with torch.no_grad():
                    param[:] = torch.Tensor(u_kernel)

        # Optionally set weights for S matrix
        if optimize_S:
            (u_out, v_out, C00_out, C11_out,
             C01_out, sigma, mu_out) = self.vampu([chi_0, chi_t])
            sigma_inv = matrix_inverse(sigma[0])
            S_nonrev = K @ sigma_inv
            S_rev = 0.5 * (S_nonrev + S_nonrev.t())
            s_kernel = np.log(np.abs(0.5 * S_rev))
            #np.save('s_kernel.npy', s_kernel)
            for param in self.vamps.parameters():
                with torch.no_grad():
                    param[:] =  torch.Tensor(s_kernel)
        # Project training and validation data with new weights
        # chi_data = DataLoader(train_data, batch_size=len(dataset), shuffle=True)
        #
        # return chi_data

    def train_US(self, data, lr_rate=1e-3, train_u = True, out_log=False):

        self.lobe.requires_grad_(False)
        self.lobe_timelagged.requires_grad_(False)
        # self.lobe.eval()
        # self.lobe_timelagged.eval()
        if train_u:
            self._vampu.train()
            self._vampu.requires_grad_(True)
            self._vampu.u_kernel.retain_grad()
        else:
            self._vampu.requires_grad_(False)
            # self._vampu.eval()


        self._vamps.train()
        self._vamps.s_kernel.retain_grad()

        self.optimizer.zero_grad()
        x_0, x_t = data[0], data[1]
        x_0 = torch.Tensor(x_0).to(device)
        x_t = torch.Tensor(x_t).to(device)


        (u_out, v_out, C00_out, C11_out,
         C01_out, sigma_out, mu_out) = self._vampu([x_0, x_t])
        Ve_out, K_out, p_out, S_out = self._vamps([
            x_0, x_t, u_out, v_out, C00_out,
            C11_out, C01_out, sigma_out])
        # (u_out, v_out, C00_out, C11_out,
        #    C01_out, sigma_out) = self._vampu([x_0, x_t])
        # Ve_out = self._vamps([
        #     x_0, x_t, u_out, v_out, C00_out,
        #     C11_out, C01_out, sigma_out])
        self._K = K_out[0]
        loss_value = vampnet_loss(Ve_out, Ve_out, method=self.score_method, epsilon=self.epsilon, mode=self.score_mode)
        loss_value.backward()
        self.optimizer.step()
        if out_log:
             print("%f"%loss_value.item())

        self.lobe.requires_grad_(True)
        self.lobe_timelagged.requires_grad_(True)
        self.lobe.train()
        self.lobe_timelagged.train()
        if not train_u:
            self._vampu.requires_grad_(True)
            self._vampu.train()


    def estimate_koopman_op(self, trajs, tau):

        if type(trajs) == list:
            traj = np.concatenate([t[:-tau] for t in trajs], axis=0)
            traj_lag = np.concatenate([t[tau:] for t in trajs], axis=0)
        else:
            traj = trajs[:-tau]
            traj_lag = trajs[tau:]
        #print(traj.shape)
        traj = torch.Tensor(traj).to(device)
        traj_lag = torch.Tensor(traj_lag).to(device)
        (u_outd, v_outd, C00_outd, C11_outd,
         C01_outd, sigma_outd, mu_outd) = self._vampu([traj, traj_lag])
        Ve_out, K_out, p_out, S_out = self._vamps([
            traj, traj_lag, u_outd, v_outd, C00_outd,
            C11_outd, C01_outd, sigma_outd])

        k = np.array(K_out[0].detach().to('cpu'))
        #print(k)
        return k

    def set_data(self, data):
        self.data = data


    @property
    def lag(self) -> int:
        """The model lag time."""
        return self._lag

    @lag.setter
    def lag(self, lag: int):
        """
        Update the model lag time for ITS calculation.
        Parameters
        ----------
        lag
           Lag time to update the model to
        """
        self._vamps.reset_weights()
        data = self.data
        self.update_auxiliary_weights(data, optimize_u=False, optimize_S=True, reset_weights=False)

        self.train_US(data, train_u=False, out_log=True)
        for i in tqdm(range(LAG_EPOCH)):
            self.train_US(data, train_u = False)
        for i in tqdm(range(LAG_EPOCH)):
            self.train_US(data)
        self.train_US(data, out_log=True)
        print("new lag %d ok" % lag)

        self._lag = lag
        self._reestimated = True


    def estimate_koopman(self, lag: int) -> np.ndarray:
        """
        Estimates the Koopman operator for a given lag time.

        Parameters
        ----------
        lag
            Lag time to estimate at

        Returns
        -------
        koop
            Koopman operator at lag time `lag`

        """
        if lag in self._k_cache:
            return self._k_cache[lag]

        self.lag = lag
        K = np.array(self._K.detach().to('cpu'))
        self._k_cache[lag] = K
        return K

    def get_ck_test(self, traj, steps, tau):
        if type(traj) == list:
            n_states = traj[0].shape[1]
        else:
            n_states = traj.shape[1]

        predicted = np.zeros((n_states, n_states, steps))
        estimated = np.zeros((n_states, n_states, steps))

        predicted[:, :, 0] = np.identity(n_states)
        estimated[:, :, 0] = np.identity(n_states)

        for vector, i in zip(np.identity(n_states), range(n_states)):
            for n in range(1, steps):
                koop = self.estimate_koopman_op(traj, tau)
                koop_pred = np.linalg.matrix_power(koop, n)
                koop_est = self.estimate_koopman_op(traj, tau*n)

                predicted[i,:,n] = vector@koop_pred
                estimated[i,:,n] = vector@koop_est

        return [predicted, estimated]

    def get_its(self, traj, lags, dt: float=1.0):
        """
        Calculate implied timescales for a sequence of lag times.
        @param lags:
        @return:
        """

        its = np.empty((self.n_output - 1, len(lags)))
        for i, lag in enumerate(lags):
            K = self.estimate_koopman_op(traj, lag)
            lambdas = np.linalg.eig(np.real(K))
            lambdas = np.sort(np.abs(np.real(lambdas)))[:LAST]
            its[:, i] = -lag * dt / np.log(lambdas)

        self.reset_lag()
        return its


    def ck_test(self, steps, tau, n_states=args.num_classes):
        """

        Perform the Chapman-Kolmogorov test on the model.
        ------
        @param traj:
        @param steps:
        @param tau:
        -----
        @return:
        """
        predicted = np.zeros((n_states, n_states, steps))
        estimated = np.zeros((n_states, n_states, steps))

        predicted[:, :, 0] = np.identity(n_states)
        estimated[:, :, 0] = np.identity(n_states)
        # Get the current Koopman operator first, because
        # estimating at a new lag time is very expensive
        self.lag = tau
        # K = self.K
        # print(K.shape)
        # Estimate new operators (slow)
        temp_est = np.empty((steps, n_states, n_states))
        # temp_est[1] = K
        for nn in range(1, steps):
            temp_est[nn] = self.estimate_koopman(tau * nn)
        self.reset_lag()
        K = temp_est[1]

        # Get new predictions (fast)
        for i in range(n_states):
            vec = np.eye(n_states)[i]
            print(vec.shape)
            for nn in range(1, steps):
                estimated[i, :, nn] = vec @ temp_est[nn]
                predicted[i, :, nn] = vec @ np.linalg.matrix_power(K, nn)
        # # Estimate new operators (slow)
        # for vector, i in zip(np.identity(n_states), range(n_states)):
        #     for n in range(1, steps):
        #         koop = K
        #         koop_pred = np.linalg.matrix_power(koop, n)
        #         koop_est = self.estimate_koopman(tau * n)
        #
        #         predicted[i, :, n] = vector @ koop_pred
        #         estimated[i, :, n] = vector @ koop_est

        return [predicted, estimated]

    def reset_lag(self):
        """Reset the model to the original lag time."""
        self.lag = self.network_lag

    def its(self, lags, dt: float=1.0):
        """
        Calculate implied timescales for a sequence of lag times.
        @param lags:
        @return:
        """

        its = np.empty((self.n_output - 1, len(lags)))
        for i, lag in enumerate(lags):
            K = self.estimate_koopman(lag)
            # lambdas = np.linalg.eig(np.real(K))
            # lambdas = np.sort(np.absolute(lambdas))[:LAST]

            k_eigvals, k_eigvec = np.linalg.eig(np.real(K))
            k_eigvals = np.sort(np.absolute(k_eigvals))
            k_eigvals = k_eigvals[:-1]
            its[:, i] = -lag * dt / np.log(k_eigvals)
            # lambdas = np.linalg.eigvals(K)
            # lambdas = np.sort(np.abs(np.real(lambdas)))[:LAST]
            # its[:, i] = -lag * dt / np.log(lambdas)

        self.reset_lag()
        return its
