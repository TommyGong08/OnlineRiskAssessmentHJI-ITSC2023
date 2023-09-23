import os
import math
import sys

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as Func
from torch.nn import init
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from model.components import *
from model.model_utils import *
import model.dynamics as dynamic_module

import torch.optim as optim


class ConvTemporalGraphical(nn.Module):
    # Source : https://github.com/yysijie/st-gcn/blob/master/net/st_gcn.py
    r"""The basic module for applying a graph convolution.
    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int): Size of the graph convolving kernel
        t_kernel_size (int): Size of the temporal convolving kernel
        t_stride (int, optional): Stride of the temporal convolution. Default: 1
        t_padding (int, optional): Temporal zero-padding added to both sides of
            the input. Default: 0
        t_dilation (int, optional): Spacing between temporal kernel elements.
            Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output.
            Default: ``True``
    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format
        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 t_kernel_size=1,
                 t_stride=1,
                 t_padding=0,
                 t_dilation=1,
                 bias=True):
        super(ConvTemporalGraphical, self).__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(t_kernel_size, 1),
            padding=(t_padding, 0),
            stride=(t_stride, 1),
            dilation=(t_dilation, 1),
            bias=bias)

    def forward(self, x, A):
        # print(A.size(0))
        assert A.size(0) == self.kernel_size
        x = self.conv(x)
        x = torch.einsum('nctv,tvw->nctw', (x, A))
        return x.contiguous(), A


class st_gcn(nn.Module):
    r"""Applies a spatial temporal graph convolution over an input graph sequence.
    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (tuple): Size of the temporal convolving kernel and graph convolving kernel
        stride (int, optional): Stride of the temporal convolution. Default: 1
        dropout (int, optional): Dropout rate of the final output. Default: 0
        residual (bool, optional): If ``True``, applies a residual mechanism. Default: ``True``
    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format
        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 use_mdn=False,
                 stride=1,
                 dropout=0,
                 residual=True):
        super(st_gcn, self).__init__()

        #         print("outstg",out_channels)

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)
        self.use_mdn = use_mdn

        self.gcn = ConvTemporalGraphical(in_channels, out_channels,
                                         kernel_size[1])

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.PReLU(),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.prelu = nn.PReLU()

    def forward(self, x, A):

        res = self.residual(x)
        x, A = self.gcn(x, A)

        x = self.tcn(x) + res

        if not self.use_mdn:
            x = self.prelu(x)

        return x, A


class Social_stgcnn(nn.Module):
    def __init__(self, device=None, log_writer=None, n_stgcnn=1, n_txpcnn=5, input_feat=2, output_feat=6,
                 seq_len=8, pred_seq_len=6, kernel_size=3):
        super(Social_stgcnn, self).__init__()
        self.device = device
        self.n_stgcnn = n_stgcnn
        self.n_txpcnn = n_txpcnn
        self.log_writer = log_writer

        self.st_gcns = nn.ModuleList()
        self.st_gcns.append(st_gcn(input_feat, output_feat, (kernel_size, seq_len)))
        for j in range(1, self.n_stgcnn):
            self.st_gcns.append(st_gcn(output_feat, output_feat, (kernel_size, seq_len)))

        self.tpcnns = nn.ModuleList()
        self.tpcnns.append(nn.Conv2d(seq_len, pred_seq_len, 3, padding=1))
        for j in range(1, self.n_txpcnn):
            self.tpcnns.append(nn.Conv2d(pred_seq_len, pred_seq_len, 3, padding=1))
        self.tpcnn_ouput = nn.Conv2d(pred_seq_len, pred_seq_len, 3, padding=1)

        self.prelus = nn.ModuleList()
        for j in range(self.n_txpcnn):
            self.prelus.append(nn.PReLU())

    def forward(self, v_raw, a, V_trgt):
        v = v_raw.permute(0, 3, 1, 2)
        for k in range(self.n_stgcnn):
            v, a = self.st_gcns[k](v, a)

        v = v.view(v.shape[0], v.shape[2], v.shape[1], v.shape[3])

        v = self.prelus[0](self.tpcnns[0](v))

        for k in range(1, self.n_txpcnn - 1):
            v = self.prelus[k](self.tpcnns[k](v)) + v

        v = self.tpcnn_ouput(v)
        V_pred = v.view(v.shape[0], v.shape[2], v.shape[1], v.shape[3])
        V_pred = V_pred.permute(0, 2, 3, 1)
        # V_pred size; (1, 6, 3, 6)
        agents_num = V_pred.size(2)
        Tra_pred = V_trgt.clone()

        for i in range(agents_num):
            x_input = v_raw[:, 7, i, :]  # current position
            log_pis = V_pred[:, :, i, 0]  # [1, 1, 6]
            mus = V_pred[:, :, i, 1:3]  # [1, 1, 6*2]
            log_sigmas = V_pred[:, :, i, 3:5]  # [1, 1, 6*2]
            corrs = V_pred[:, :, i, 5]  # [1, 1, 6]

            log_pis = log_pis.reshape(-1, 1)
            mus = mus.reshape(-1, 2)  # [1, 1, 30]
            log_sigmas = log_sigmas.reshape(-1, 2)  # [1, 1, 30]
            corrs = corrs.reshape(-1, 1)  # [1, 1, 6]
            # a_dist = GMM2D(torch.reshape(log_pis, [num_samples, -1, ph, num_components]),
            #                torch.reshape(mus, [num_samples, -1, ph, num_components*pred_dim]),
            #                torch.reshape(log_sigmas, [num_samples, -1, ph, num_components*pred_dim]),
            #                torch.reshape(corrs, [num_samples, -1, ph, num_components]))

            a_dist = GMM2D(log_pis, mus, log_sigmas, corrs)
            a_sample = a_dist.mode()
            # dynamic model
            dt = 0.5
            result = torch.cumsum(a_sample, dim=1) * dt + x_input  # dt = 0.5
            Tra_pred[:, :, i, :] = result  # [1, 6, 3, 2]

            ysig = a_dist.get_covariance_matrix()

        return Tra_pred, a_dist, ysig

    def train_loss1(self, V_pred, V_trgt):
        V_tr = V_trgt.squeeze()
        # A_tr = A_tr.squeeze()
        V_pred = V_pred.squeeze()

        # mux, muy, sx, sy, corr
        # assert V_pred.shape == V_trgt.shape
        normx = V_trgt[:, :, 0] - V_pred[:, :, 0]
        normy = V_trgt[:, :, 1] - V_pred[:, :, 1]

        sx = torch.exp(V_pred[:, :, 2])  # sx
        sy = torch.exp(V_pred[:, :, 3])  # sy
        corr = torch.tanh(V_pred[:, :, 4])  # corr

        sxsy = sx * sy

        z = (normx / sx) ** 2 + (normy / sy) ** 2 - 2 * ((corr * normx * normy) / sxsy)
        negRho = 1 - corr ** 2

        # Numerator
        result = torch.exp(-z / (2 * negRho))
        # Normalization factor
        denom = 2 * np.pi * (sxsy * torch.sqrt(negRho))

        # Final PDF calculation
        result = result / denom

        # Numerical stability
        epsilon = 1e-20

        result = -torch.log(torch.clamp(result, min=epsilon, max=100))
        result = torch.mean(result)

        return result

    def train_loss2(self, V_pred, V_trgt):
        loss_function = torch.nn.MSELoss()
        loss = loss_function(V_trgt, V_pred)
        return loss

    def predict_actions1(self,
                         scene,
                         timesteps,
                         ph,
                         num_samples=1,
                         min_future_timesteps=0,
                         min_history_timesteps=1,
                         z_mode=False,
                         gmm_mode=False,
                         full_dist=False,
                         all_z_sep=False):

        predictions_dict = {}
        predictions_sig_dict = {}
        for node_type in self.env.NodeType:
            if node_type not in self.pred_state:
                continue

            model = self.node_models_dict[node_type]

            # Get Input data for node type and given timesteps
            batch = get_timesteps_data(env=self.env, scene=scene, t=timesteps, node_type=node_type, state=self.state,
                                       pred_state=self.pred_state, edge_types=model.edge_types,
                                       min_ht=min_history_timesteps, max_ht=self.max_ht, min_ft=min_future_timesteps,
                                       max_ft=min_future_timesteps, hyperparams=self.hyperparams, pred_model_type=1)
            # There are no nodes of type present for timestep
            if batch is None:
                continue
            (first_history_index,
             x_t, y_t, x_st_t, y_st_t,
             neighbors_data_st,
             neighbors_edge_value,
             robot_traj_st_t,
             map), nodes, timesteps_o = batch

            x = x_t.to(self.device)
            x_st_t = x_st_t.to(self.device)
            if robot_traj_st_t is not None:
                robot_traj_st_t = robot_traj_st_t.to(self.device)
            if type(map) == torch.Tensor:
                map = map.to(self.device)

            # Run forward pass

            print("debug3")
            print(x.shape)  # [2, 9, 8]
            print(x_st_t.shape)  # [2, 9, 8]
            predictions, predictions_sigma = model.predict_actions(inputs=x,
                                                                   inputs_st=x_st_t,
                                                                   first_history_indices=first_history_index,
                                                                   neighbors=neighbors_data_st,
                                                                   neighbors_edge_value=neighbors_edge_value,
                                                                   robot=robot_traj_st_t,
                                                                   map=map,
                                                                   prediction_horizon=ph,
                                                                   num_samples=num_samples,
                                                                   z_mode=z_mode,
                                                                   gmm_mode=gmm_mode,
                                                                   full_dist=full_dist,
                                                                   all_z_sep=all_z_sep, )

            predictions_np = predictions.mode().cpu().detach().numpy()  # []
            predictions_sigma_np = predictions_sigma.cpu().detach().numpy()
            print("debug4")
            print(predictions_np.shape)  # (1, 2, 6, 2)
            print(predictions_sigma_np.shape)  # (1, 2, 6, 1, 2, 2)

            # Assign predictions to node
            for i, ts in enumerate(timesteps_o):
                # print(predictions_np[:, [i]])
                if ts not in predictions_dict.keys():
                    predictions_dict[ts] = dict()
                predictions_dict[ts][nodes[i]] = np.transpose(predictions_np[:, [i]], (1, 0, 2, 3))

            for i, ts in enumerate(timesteps_o):
                if ts not in predictions_sig_dict.keys():
                    predictions_sig_dict[ts] = dict()
                predictions_sig_dict[ts][nodes[i]] = np.transpose(predictions_sigma_np[:, [i]], (1, 0, 2, 3, 4, 5))
        return predictions_dict, predictions_sig_dict

    def predict(self, v_raw, a):

        v = v_raw.permute(0, 3, 1, 2)
        print("v_raw shape", v_raw.shape)
        print("a shape", a.shape)

        for k in range(self.n_stgcnn):
            v, a = self.st_gcns[k](v, a)

        v = v.view(v.shape[0], v.shape[2], v.shape[1], v.shape[3])

        v = self.prelus[0](self.tpcnns[0](v))

        for k in range(1, self.n_txpcnn - 1):
            v = self.prelus[k](self.tpcnns[k](v)) + v

        v = self.tpcnn_ouput(v)
        V_pred = v.permute(0, 2, 1, 3)
        # V_pred = v.view(v.shape[0], v.shape[2], v.shape[1], v.shape[3])
        V_pred = V_pred.permute(0, 2, 3, 1)
        # V_pred size; (1, 6, 3, 6)
        print("V_pred:", V_pred)
        print("v_raw:", v_raw)
        agents_num = V_pred.size(2)
        seq_len = v_raw.shape[1]

        log_pis_list = []
        mus_list = []
        log_sigmas_list = []
        corrs_list = []
        print("agent number: ", agents_num)
        for i in range(agents_num):
            x_input = v_raw[:, seq_len-1, i, :]  # current position
            log_pis = V_pred[:, :, i, 0]  # [1, 1, 6]
            mus = V_pred[:, :, i, 1:3]  # [1, 1, 6*2]
            log_sigmas = V_pred[:, :, i, 3:5]  # [1, 1, 6*2]
            corrs = V_pred[:, :, i, 5]  # [1, 1, 6]

            log_pis = log_pis.reshape(-1, 1).unsqueeze(0)
            mus = mus.reshape(-1, 2).unsqueeze(0)  # [1, 1, 30]
            log_sigmas = log_sigmas.reshape(-1, 2).unsqueeze(0)  # [1, 1, 30]
            corrs = corrs.reshape(-1, 1).unsqueeze(0)  # [1, 1, 6]

            log_pis_list.append(log_pis)
            mus_list.append(mus)
            log_sigmas_list.append(log_sigmas)
            corrs_list.append(corrs)

            # print(log_pis)
            # print(mus)
            # print(log_sigmas.shape)
            # print(corrs.shape)
            # a_dist = GMM2D(torch.reshape(log_pis, [num_samples, -1, ph, num_components]),
            #                torch.reshape(mus, [num_samples, -1, ph, num_components*pred_dim]),
            #                torch.reshape(log_sigmas, [num_samples, -1, ph, num_components*pred_dim]),
            #                torch.reshape(corrs, [num_samples, -1, ph, num_components]))

        out_log_pis = torch.stack(log_pis_list, dim=1)
        out_mus = torch.stack(mus_list, dim=1)
        out_log_sigmas = torch.stack(log_sigmas_list, dim=1)
        out_corrs = torch.stack(corrs_list, dim=1)

        print("debug5")
        print("log_pi_t shape", log_pis.shape) # [2, 6, 1]
        print("mu_t shape", mus.shape)  # [2, 6, 2]
        print("log_sigma_t shape", log_sigmas.shape) # [2, 6, 2]
        print("corr_t shape", corrs.shape) # [2, 6, 1]

        a_dist = GMM2D(out_log_pis, out_mus, out_log_sigmas, out_corrs)
        a_sample = a_dist.mode()
        print("a_dist:", a_sample.shape)  # (6, 2)
        # dynamic model
        dt = 0.5
        # result = torch.cumsum(a_sample, dim=1) * dt + x_input  # dt = 0.5

        ysig = a_dist.get_covariance_matrix()

        print("a_dist:", a_sample.shape)  # (6, 2)
        print("ysig:", ysig.shape)  # (6, 1, 2, 2)


        return a_dist, ysig









