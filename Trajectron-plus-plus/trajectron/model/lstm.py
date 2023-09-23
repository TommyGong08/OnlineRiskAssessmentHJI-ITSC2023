import torch
import torch.nn as nn
from model.components import *

class LstmRNN(nn.Module):
    """
        Parameters：
        - input_size: feature size
        - hidden_size: number of hidden units
        - output_size: number of output
        - num_layers: layers of LSTM to stack
    """

    def __init__(self, device=None, log_writer=None, input_size=2, hidden_size=32, output_size=6, num_layers=3,
                 seq_len=8, pred_seq_len=6, n_txpcnn=5):
        super().__init__()

        self.device = device
        self.n_txpcnn = n_txpcnn
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)  # utilize the LSTM model in torch.nn
        self.linear1 = nn.Linear(hidden_size, output_size)  #

        self.tpcnns = nn.ModuleList()
        self.tpcnns.append(nn.Conv2d(seq_len, pred_seq_len, 3, padding=1))
        for j in range(1, self.n_txpcnn):
            self.tpcnns.append(nn.Conv2d(pred_seq_len, pred_seq_len, 3, padding=1))
        self.tpcnn_ouput = nn.Conv2d(pred_seq_len, pred_seq_len, 3, padding=1)

        self.prelus = nn.ModuleList()
        for j in range(self.n_txpcnn):
            self.prelus.append(nn.PReLU())

    def forward(self, v_raw, a, V_trgt):
        x = v_raw.squeeze(0)  # [8, 3, 2]
        # _x = v_raw.permute(1, 0, 2)  # [3, 8, 2]
        x, _ = self.lstm(x)  # _x is input, size (seq_len, batch, input_size)
        s, b, h = x.shape  # x is output, size (seq_len, batch, hidden_size)
        x = x.view(s * b, h)
        x = self.linear1(x)
        x = x.view(s, b, -1)  # [ 3, 8, 5]
        x = x.unsqueeze(0)
        x = x.permute(0, 1, 3, 2)
        v = self.prelus[0](self.tpcnns[0](x))

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
            Tra_pred[:, :, i, :] = result  # [1, 6, n, 2]

            a_cov = a_dist.get_covariance_matrix()

        # print("a_dist:", a_sample.shape)  # (6, 2)
        # print("ysig:", ysig.shape)  # (6, 1, 2, 2)
        return Tra_pred, a_sample, a_cov

    def train_loss2(self, V_pred, V_trgt):
        loss_function = torch.nn.MSELoss()
        loss = loss_function(V_trgt, V_pred)
        return loss

    def predict(self, v_raw, a):
        x = v_raw.squeeze(0)  # [8, 3, 2]
        # _x = v_raw.permute(1, 0, 2)  # [3, 8, 2]
        x, _ = self.lstm(x)  # _x is input, size (seq_len, batch, input_size)
        s, b, h = x.shape  # x is output, size (seq_len, batch, hidden_size)
        x = x.view(s * b, h)
        x = self.linear1(x)
        x = x.view(s, b, -1)  # [ 3, 8, 5]
        x = x.unsqueeze(0)
        v = x.permute(0, 1, 3, 2)

        v = self.prelus[0](self.tpcnns[0](v))

        for k in range(1, self.n_txpcnn - 1):
            v = self.prelus[k](self.tpcnns[k](v)) + v

        v = self.tpcnn_ouput(v)
        V_pred = v.view(v.shape[0], v.shape[2], v.shape[1], v.shape[3])
        V_pred = V_pred.permute(0, 2, 3, 1)
        # V_pred size; (1, 6, 3, 6)
        agents_num = V_pred.size(2)

        log_pis_list = []
        mus_list = []
        log_sigmas_list = []
        corrs_list = []

        for i in range(agents_num):
            x_input = v_raw[:, 7, i, :]  # current position
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

            print(log_pis.shape)
            print(mus.shape)
            print(log_sigmas.shape)
            print(corrs.shape)
            # a_dist = GMM2D(torch.reshape(log_pis, [num_samples, -1, ph, num_components]),
            #                torch.reshape(mus, [num_samples, -1, ph, num_components*pred_dim]),
            #                torch.reshape(log_sigmas, [num_samples, -1, ph, num_components*pred_dim]),
            #                torch.reshape(corrs, [num_samples, -1, ph, num_components]))

        out_log_pis = torch.stack(log_pis_list, dim=1)
        out_mus = torch.stack(mus_list, dim=1)
        out_log_sigmas = torch.stack(log_sigmas_list, dim=1)
        out_corrs = torch.stack(corrs_list, dim=1)

        print("debug5")
        print("log_pi_t shape", log_pis)  # [2, 6, 1]
        print("mu_t shape", mus)  # [2, 6, 2]
        print("log_sigma_t shape", log_sigmas)  # [2, 6, 2]
        print("corr_t shape", corrs)  # [2, 6, 1]

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


if __name__ == '__main__':
    model = LstmRNN()
    v_obs = torch.randn(1, 8, 3, 2)
    A_obs = torch.randn(1, 8, 3, 3)
    v_target = torch.randn(1, 6, 3, 2)
    output, a_sample, a_cov = model(v_obs, A_obs, v_target)




