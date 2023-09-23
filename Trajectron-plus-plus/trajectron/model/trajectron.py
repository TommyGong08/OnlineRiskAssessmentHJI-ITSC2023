import torch
import numpy as np
from .mgcvae import MultimodalGenerativeCVAE
from .dataset import get_timesteps_data, restore, get_timesteps_data_stgcnn

class Trajectron(object):
    def __init__(self, model_registrar,
                 hyperparams, log_writer,
                 device):
        super(Trajectron, self).__init__()
        self.hyperparams = hyperparams
        self.log_writer = log_writer
        self.device = device
        self.curr_iter = 0

        self.model_registrar = model_registrar
        self.node_models_dict = dict()
        self.nodes = set()

        self.env = None

        self.min_ht = self.hyperparams['minimum_history_length']
        self.max_ht = self.hyperparams['maximum_history_length']
        self.ph = self.hyperparams['prediction_horizon']
        self.state = self.hyperparams['state']
        self.state_length = dict()
        for state_type in self.state.keys():
            self.state_length[state_type] = int(
                np.sum([len(entity_dims) for entity_dims in self.state[state_type].values()])
            )
        self.pred_state = self.hyperparams['pred_state']

    def set_environment(self, env):
        self.env = env

        self.node_models_dict.clear()
        edge_types = env.get_edge_types()

        for node_type in env.NodeType:
            # Only add a Model for NodeTypes we want to predict
            if node_type in self.pred_state.keys():
                self.node_models_dict[node_type] = MultimodalGenerativeCVAE(env,
                                                                            node_type,
                                                                            self.model_registrar,
                                                                            self.hyperparams,
                                                                            self.device,
                                                                            edge_types,
                                                                            log_writer=self.log_writer)

    def set_curr_iter(self, curr_iter):
        self.curr_iter = curr_iter
        for node_str, model in self.node_models_dict.items():
            model.set_curr_iter(curr_iter)

    def set_annealing_params(self):
        for node_str, model in self.node_models_dict.items():
            model.set_annealing_params()

    def step_annealers(self, node_type=None):
        if node_type is None:
            for node_type in self.node_models_dict:
                self.node_models_dict[node_type].step_annealers()
        else:
            self.node_models_dict[node_type].step_annealers()

    def train_loss(self, batch, node_type):
        (first_history_index,
         x_t, y_t, x_st_t, y_st_t,
         neighbors_data_st,
         neighbors_edge_value,
         robot_traj_st_t,
         map) = batch

        x = x_t.to(self.device)
        y = y_t.to(self.device)
        x_st_t = x_st_t.to(self.device)
        y_st_t = y_st_t.to(self.device)
        if robot_traj_st_t is not None:
            robot_traj_st_t = robot_traj_st_t.to(self.device)
        if type(map) == torch.Tensor:
            map = map.to(self.device)

        # Run forward pass
        model = self.node_models_dict[node_type]
        loss = model.train_loss(inputs=x,
                                inputs_st=x_st_t,
                                first_history_indices=first_history_index,
                                labels=y,
                                labels_st=y_st_t,
                                neighbors=restore(neighbors_data_st),
                                neighbors_edge_value=restore(neighbors_edge_value),
                                robot=robot_traj_st_t,
                                map=map,
                                prediction_horizon=self.ph)

        return loss

    def eval_loss(self, batch, node_type):
        (first_history_index,
         x_t, y_t, x_st_t, y_st_t,
         neighbors_data_st,
         neighbors_edge_value,
         robot_traj_st_t,
         map) = batch

        x = x_t.to(self.device)
        y = y_t.to(self.device)
        x_st_t = x_st_t.to(self.device)
        y_st_t = y_st_t.to(self.device)
        if robot_traj_st_t is not None:
            robot_traj_st_t = robot_traj_st_t.to(self.device)
        if type(map) == torch.Tensor:
            map = map.to(self.device)

        # Run forward pass
        model = self.node_models_dict[node_type]
        nll = model.eval_loss(inputs=x,
                              inputs_st=x_st_t,
                              first_history_indices=first_history_index,
                              labels=y,
                              labels_st=y_st_t,
                              neighbors=restore(neighbors_data_st),
                              neighbors_edge_value=restore(neighbors_edge_value),
                              robot=robot_traj_st_t,
                              map=map,
                              prediction_horizon=self.ph)

        return nll.cpu().detach().numpy()

    def predict(self,
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
        for node_type in self.env.NodeType:
            if node_type not in self.pred_state:
                continue

            model = self.node_models_dict[node_type]

            # Get Input data for node type and given timesteps
            batch = get_timesteps_data(env=self.env, scene=scene, t=timesteps, node_type=node_type, state=self.state,
                                       pred_state=self.pred_state, edge_types=model.edge_types,
                                       min_ht=min_history_timesteps, max_ht=self.max_ht, min_ft=min_future_timesteps,
                                       max_ft=min_future_timesteps, hyperparams=self.hyperparams)
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
            predictions = model.predict(inputs=x,
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
                                        all_z_sep=all_z_sep)

            predictions_np = predictions.cpu().detach().numpy()

            # Assign predictions to node
            for i, ts in enumerate(timesteps_o):
                if ts not in predictions_dict.keys():
                    predictions_dict[ts] = dict()
                predictions_dict[ts][nodes[i]] = np.transpose(predictions_np[:, [i]], (1, 0, 2, 3))

        return predictions_dict

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

            # print("debug3")
            # print(x.shape)  # [2, 9, 8]
            # print(x_st_t.shape) # [2, 9, 8]

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
                                        all_z_sep=all_z_sep,)

            predictions_np = predictions.mode().cpu().detach().numpy()  # []
            predictions_sigma_np = predictions_sigma.cpu().detach().numpy()
            # print("debug4")
            # print(predictions_np.shape)   # (1, 2, 6, 2)
            # print(predictions_sigma_np.shape)  # (1, 2, 6, 1, 2, 2)
            # print(timesteps_o)  # [35, 35]

            if timesteps == 10:
                print("###timestep")
                print(x_t)
                motion_pred_data_input_npy = []
                motion_pred_data_input_npy = motion_pred_data_input_npy.append(x_t)
                motion_pred_data_input_npy = np.array(motion_pred_data_input_npy)
                np.save("./data/pred_input_trajectron++.npy", motion_pred_data_input_npy)

                print(predictions_np)
                motion_pred_data_output_npy = []
                motion_pred_data_output_npy = motion_pred_data_output_npy.append(predictions_np)
                np.save("./data/pred_output_trajectron++.npy", motion_pred_data_output_npy)

            # Assign predictions to node
            for i, ts in enumerate(timesteps_o):
                # print("i:", i)
                # print("ts: ", ts)
                # print(predictions_np[:, [i]])

                if ts not in predictions_dict.keys():
                    predictions_dict[ts] = dict()
                predictions_dict[ts][nodes[i]] = np.transpose(predictions_np[:, [i]], (1, 0, 2, 3))
            
            for i, ts in enumerate(timesteps_o):
                if ts not in predictions_sig_dict.keys():
                    predictions_sig_dict[ts] = dict()
                predictions_sig_dict[ts][nodes[i]] = np.transpose(predictions_sigma_np[:, [i]], (1, 0, 2, 3, 4, 5))
        return predictions_dict, predictions_sig_dict

    def predict_actions2(self,
                        scene,
                        timesteps,
                        ph,
                        num_samples=1,
                        min_future_timesteps=0,
                        min_history_timesteps=1,
                        z_mode=False,
                        gmm_mode=False,
                        full_dist=False,
                        all_z_sep=False,
                        ):

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
                                       max_ft=min_future_timesteps, hyperparams=self.hyperparams, pred_model_type=2)
            # There are no nodes of type present for timestep
            if batch is None:
                continue
            (first_history_index,
             x_t, y_t, x_st_t, y_st_t,
             neighbors_data_st,
             neighbors_edge_value,
             robot_traj_st_t,
             map, V_obs, A_obs, V_tr, A_tr), nodes, timesteps_o = batch

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
            # predictions, predictions_sigma = model.predict_actions(inputs=x,
            #                                                        inputs_st=x_st_t,
            #                                                        first_history_indices=first_history_index,
            #                                                        neighbors=neighbors_data_st,
            #                                                        neighbors_edge_value=neighbors_edge_value,
            #                                                        robot=robot_traj_st_t,
            #                                                        map=map,
            #                                                        prediction_horizon=ph,
            #                                                        num_samples=num_samples,
            #                                                        z_mode=z_mode,
            #                                                        gmm_mode=gmm_mode,
            #                                                        full_dist=full_dist,
            #                                                        all_z_sep=all_z_sep)

            predictions, predictions_sigma = model(V_obs, A_obs.squeeze(), V_tr)

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

    def predict_actions_stgcnn(self,
                         scene,
                         timesteps,
                         ph,
                         num_samples=1,
                         min_future_timesteps=0,
                         min_history_timesteps=1,
                         z_mode=False,
                         gmm_mode=False,
                         full_dist=False,
                         all_z_sep=False,
                         pred_model=None):

        predictions_dict = {}
        predictions_sig_dict = {}
        for node_type in self.env.NodeType:
            if node_type not in self.pred_state:
                continue

            model = self.node_models_dict[node_type]

            # Get Input data for node type and given timesteps
            batch = get_timesteps_data_stgcnn(env=self.env, scene=scene, t=timesteps, node_type=node_type, state=self.state,
                                           pred_state=self.pred_state, edge_types=model.edge_types,
                                           min_ht=min_history_timesteps, max_ht=self.max_ht, min_ft=min_future_timesteps,
                                           max_ft=min_future_timesteps, hyperparams=self.hyperparams, pred_model_type=2)
            # batch = [tensor.cuda() for tensor in batch]
            # There are no nodes of type present for timestep
            if batch is None:
                continue

            (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel,
             non_linear_ped, loss_mask, V_obs, A_obs, V_tr, A_tr), nodes, timesteps_o = batch

            # (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel,
            #  non_linear_ped, loss_mask, V_obs, A_obs), nodes, timesteps_o = batch

            # x = x_t.to(self.device)
            # x_st_t = x_st_t.to(self.device)
            # if robot_traj_st_t is not None:
            #     robot_traj_st_t = robot_traj_st_t.to(self.device)
            # if type(map) == torch.Tensor:
            #     map = map.to(self.device)

            # Run forward pass

            print("debug31")
            V_obs = V_obs.unsqueeze(0)
            print(V_obs.shape)  # [1, 8, 2, 2]
            print(A_obs.shape)  # [8, 2, 2]

            predictions, predictions_sigma = pred_model.predict(V_obs, A_obs)

            # predictions, predictions_sigma = model.predict_actions(inputs=x,
            #                                                        inputs_st=x_st_t,
            #                                                        first_history_indices=first_history_index,
            #                                                        neighbors=neighbors_data_st,
            #                                                        neighbors_edge_value=neighbors_edge_value,
            #                                                        robot=robot_traj_st_t,
            #                                                        map=map,
            #                                                        prediction_horizon=ph,
            #                                                        num_samples=num_samples,
            #                                                        z_mode=z_mode,
            #                                                        gmm_mode=gmm_mode,
            #                                                        full_dist=full_dist,
            #                                                        all_z_sep=all_z_sep, )

            predictions_np = predictions.mode().cpu().detach().numpy()  # []
            predictions_sigma_np = predictions_sigma.cpu().detach().numpy()

            if timesteps == 10:
                print("###timestep_stgcnn")
                # print(x_t)
                # motion_pred_data_input_npy = []
                # motion_pred_data_input_npy = motion_pred_data_input_npy.append(x_t)
                # motion_pred_data_input_npy = np.array(motion_pred_data_input_npy)
                # np.save("./data/pred_input_trajectron++.npy", motion_pred_data_input_npy)

                print(predictions_np)
                motion_pred_data_output_npy = []
                motion_pred_data_output_npy = motion_pred_data_output_npy.append(predictions_np)
                np.save("./data/pred_output_stgcnn.npy", motion_pred_data_output_npy)


            # print("debug4")
            # print(predictions_np.shape)  # (1, 2, 6, 2)
            # print(predictions_sigma_np.shape)  # (1, 2, 6, 1, 2, 2)
            # print(timesteps_o)
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
