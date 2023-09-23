import torch
import numpy as np
import collections.abc
from torch.utils.data._utils.collate import default_collate
import dill
import math
import torch
import networkx as nx
import tqdm
from utils import trajectory_utils
container_abcs = collections.abc


def poly_fit(traj, traj_len, threshold):
    """
    Input:
    - traj: Numpy array of shape (2, traj_len)
    - traj_len: Len of trajectory
    - threshold: Minimum error to be considered for non linear traj
    Output:
    - int: 1 -> Non Linear 0-> Linear
    """
    t = np.linspace(0, traj_len - 1, traj_len)
    res_x = np.polyfit(t, traj[0, -traj_len:], 2, full=True)[1]
    res_y = np.polyfit(t, traj[1, -traj_len:], 2, full=True)[1]
    if res_x + res_y >= threshold:
        return 1.0
    else:
        return 0.0


def anorm(p1, p2):
    NORM = math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
    if NORM == 0:
        return 0
    return 1 / (NORM)


def seq_to_graph(seq_, seq_rel, norm_lap_matr=True):
    seq_ = seq_.squeeze()
    seq_rel = seq_rel.squeeze()
    seq_len = seq_.shape[2]
    max_nodes = seq_.shape[0]

    # V shape:(sequance length, nodes num, feature num)
    V = np.zeros((seq_len, max_nodes, 2))
    A = np.zeros((seq_len, max_nodes, max_nodes))
    for s in range(seq_len):
        step_ = seq_[:, :, s]
        step_rel = seq_rel[:, :, s]
        for h in range(len(step_)):
            V[s, h, :] = step_rel[h]
            A[s, h, h] = 1
            for k in range(h + 1, len(step_)):
                l2_norm = anorm(step_rel[h], step_rel[k])
                A[s, h, k] = l2_norm
                A[s, k, h] = l2_norm
        if norm_lap_matr:
            G = nx.from_numpy_matrix(A[s, :, :])
            A[s, :, :] = nx.normalized_laplacian_matrix(G).toarray()

    return torch.from_numpy(V).type(torch.float), \
        torch.from_numpy(A).type(torch.float)

def restore(data):
    """
    In case we dilled some structures to share between multiple process this function will restore them.
    If the data input are not bytes we assume it was not dilled in the first place

    :param data: Possibly dilled data structure
    :return: Un-dilled data structure
    """
    if type(data) is bytes:
        return dill.loads(data)
    return data


def collate(batch):
    if len(batch) == 0:
        return batch
    elem = batch[0]
    if elem is None:
        return None
    elif isinstance(elem, container_abcs.Sequence):
        if len(elem) == 4: # We assume those are the maps, map points, headings and patch_size
            scene_map, scene_pts, heading_angle, patch_size = zip(*batch)
            if heading_angle[0] is None:
                heading_angle = None
            else:
                heading_angle = torch.Tensor(heading_angle)
            map = scene_map[0].get_cropped_maps_from_scene_map_batch(scene_map,
                                                                     scene_pts=torch.Tensor(scene_pts),
                                                                     patch_size=patch_size[0],
                                                                     rotation=heading_angle)
            return map
        transposed = zip(*batch)
        return [collate(samples) for samples in transposed]
    elif isinstance(elem, container_abcs.Mapping):
        # We have to dill the neighbors structures. Otherwise each tensor is put into
        # shared memory separately -> slow, file pointer overhead
        # we only do this in multiprocessing
        neighbor_dict = {key: [d[key] for d in batch] for key in elem}
        return dill.dumps(neighbor_dict) if torch.utils.data.get_worker_info() else neighbor_dict
    return default_collate(batch)


def get_relative_robot_traj(env, state, node_traj, robot_traj, node_type, robot_type):
    # TODO: We will have to make this more generic if robot_type != node_type
    # Make Robot State relative to node
    _, std = env.get_standardize_params(state[robot_type], node_type=robot_type)
    std[0:2] = env.attention_radius[(node_type, robot_type)]
    robot_traj_st = env.standardize(robot_traj,
                                    state[robot_type],
                                    node_type=robot_type,
                                    mean=node_traj,
                                    std=std)
    robot_traj_st_t = torch.tensor(robot_traj_st, dtype=torch.float)

    return robot_traj_st_t


def get_node_timestep_data(env, scene, t, node, state, pred_state,
                           edge_types, max_ht, max_ft, hyperparams,
                           scene_graph=None, pred_model_type=2):
    """
    Pre-processes the data for a single batch element: node state over time for a specific time in a specific scene
    as well as the neighbour data for it.

    :param env: Environment
    :param scene: Scene
    :param t: Timestep in scene
    :param node: Node
    :param state: Specification of the node state
    :param pred_state: Specification of the prediction state
    :param edge_types: List of all Edge Types for which neighbours are pre-processed
    :param max_ht: Maximum history timesteps
    :param max_ft: Maximum future timesteps (prediction horizon)
    :param hyperparams: Model hyperparameters
    :param scene_graph: If scene graph was already computed for this scene and time you can pass it here
    :return: Batch Element

    Args:
        pred_model_type:
        pred_model_type:
    """

    # Node
    timestep_range_x = np.array([t - max_ht, t])
    timestep_range_y = np.array([t + 1, t + max_ft])

    x = node.get(timestep_range_x, state[node.type])
    y = node.get(timestep_range_y, pred_state[node.type])
    first_history_index = (max_ht - node.history_points_at(t)).clip(0)

    _, std = env.get_standardize_params(state[node.type], node.type)
    std[0:2] = env.attention_radius[(node.type, node.type)]
    rel_state = np.zeros_like(x[0])
    rel_state[0:2] = np.array(x)[-1, 0:2]
    x_st = env.standardize(x, state[node.type], node.type, mean=rel_state, std=std)
    if list(pred_state[node.type].keys())[0] == 'position':  # If we predict position we do it relative to current pos
        y_st = env.standardize(y, pred_state[node.type], node.type, mean=rel_state[0:2])
    else:
        y_st = env.standardize(y, pred_state[node.type], node.type)

    if pred_model_type == 1:
        x_t = torch.tensor(x, dtype=torch.float)
        y_t = torch.tensor(y, dtype=torch.float)
    else:
        x_t = torch.tensor(x, dtype=torch.float).permute(1, 0)
        y_t = torch.tensor(y, dtype=torch.float).permute(1, 0)

    # x_t = torch.tensor(x, dtype=torch.float)
    # y_t = torch.tensor(y, dtype=torch.float)
    x_st_t = torch.tensor(x_st, dtype=torch.float)
    y_st_t = torch.tensor(y_st, dtype=torch.float)

    if pred_model_type != 1:
        V_x, A_x, V_y, A_y = [], [], [], []
        v_, a_ = trajectory_utils.seq_to_graph(x_t[:, 0:2], x_st_t[:, 0:2])
        V_x.append(v_.clone())
        A_x.append(a_.clone())
        # v_, a_ = trajectory_utils.seq_to_graph(y_t[:, 0:2], y_st_t[:, 0:2])
        # V_y.append(v_.clone())
        # A_y.append(a_.clone())

    # Neighbors
    neighbors_data_st = None
    neighbors_edge_value = None
    if hyperparams['edge_encoding']:
        # Scene Graph
        scene_graph = scene.get_scene_graph(t,
                                            env.attention_radius,
                                            hyperparams['edge_addition_filter'],
                                            hyperparams['edge_removal_filter']) if scene_graph is None else scene_graph

        neighbors_data_st = dict()
        neighbors_edge_value = dict()
        for edge_type in edge_types:
            neighbors_data_st[edge_type] = list()
            # We get all nodes which are connected to the current node for the current timestep
            connected_nodes = scene_graph.get_neighbors(node, edge_type[1])

            if hyperparams['dynamic_edges'] == 'yes':
                # We get the edge masks for the current node at the current timestep
                edge_masks = torch.tensor(scene_graph.get_edge_scaling(node), dtype=torch.float)
                neighbors_edge_value[edge_type] = edge_masks

            for connected_node in connected_nodes:
                neighbor_state_np = connected_node.get(np.array([t - max_ht, t]),
                                                       state[connected_node.type],
                                                       padding=0.0)

                # Make State relative to node where neighbor and node have same state
                _, std = env.get_standardize_params(state[connected_node.type], node_type=connected_node.type)
                std[0:2] = env.attention_radius[edge_type]
                equal_dims = np.min((neighbor_state_np.shape[-1], x.shape[-1]))
                rel_state = np.zeros_like(neighbor_state_np)
                rel_state[:, ..., :equal_dims] = x[-1, ..., :equal_dims]
                neighbor_state_np_st = env.standardize(neighbor_state_np,
                                                       state[connected_node.type],
                                                       node_type=connected_node.type,
                                                       mean=rel_state,
                                                       std=std)

                neighbor_state = torch.tensor(neighbor_state_np_st, dtype=torch.float)
                neighbors_data_st[edge_type].append(neighbor_state)

    # Robot
    robot_traj_st_t = None
    if hyperparams['incl_robot_node']:
        timestep_range_r = np.array([t, t + max_ft])
        if scene.non_aug_scene is not None:
            robot = scene.get_node_by_id(scene.non_aug_scene.robot.id)
        else:
            robot = scene.robot
        robot_type = robot.type
        robot_traj = robot.get(timestep_range_r, state[robot_type], padding=0.0)
        node_state = np.zeros_like(robot_traj[0])
        node_state[:x.shape[1]] = x[-1]
        robot_traj_st_t = get_relative_robot_traj(env, state, node_state, robot_traj, node.type, robot_type)

    # Map
    map_tuple = None
    if hyperparams['use_map_encoding']:
        if node.type in hyperparams['map_encoder']:
            if node.non_aug_node is not None:
                x = node.non_aug_node.get(np.array([t]), state[node.type])
            me_hyp = hyperparams['map_encoder'][node.type]
            if 'heading_state_index' in me_hyp:
                heading_state_index = me_hyp['heading_state_index']
                # We have to rotate the map in the opposit direction of the agent to match them
                if type(heading_state_index) is list:  # infer from velocity or heading vector
                    heading_angle = -np.arctan2(x[-1, heading_state_index[1]],
                                                x[-1, heading_state_index[0]]) * 180 / np.pi
                else:
                    heading_angle = -x[-1, heading_state_index] * 180 / np.pi
            else:
                heading_angle = None

            scene_map = scene.map[node.type]
            map_point = x[-1, :2]


            patch_size = hyperparams['map_encoder'][node.type]['patch_size']
            map_tuple = (scene_map, map_point, heading_angle, patch_size)
    if pred_model_type > 1:
        return (first_history_index, x_t, y_t, x_st_t, y_st_t, neighbors_data_st,
            neighbors_edge_value, robot_traj_st_t, map_tuple, V_x, A_x)
    else:
        return (first_history_index, x_t, y_t, x_st_t, y_st_t, neighbors_data_st,
                neighbors_edge_value, robot_traj_st_t, map_tuple)


def get_timesteps_data(env, scene, t, node_type, state, pred_state,
                       edge_types, min_ht, max_ht, min_ft, max_ft, hyperparams, pred_model_type=2):
    """
    Puts together the inputs for ALL nodes in a given scene and timestep in it.

    :param env: Environment
    :param scene: Scene
    :param t: Timestep in scene
    :param node_type: Node Type of nodes for which the data shall be pre-processed
    :param state: Specification of the node state
    :param pred_state: Specification of the prediction state
    :param edge_types: List of all Edge Types for which neighbors are pre-processed
    :param max_ht: Maximum history timesteps
    :param max_ft: Maximum future timesteps (prediction horizon)
    :param hyperparams: Model hyperparameters
    :return:
    """
    nodes_per_ts = scene.present_nodes(t,
                                       type=node_type,
                                       min_history_timesteps=min_ht,
                                       min_future_timesteps=max_ft,
                                       return_robot=not hyperparams['incl_robot_node'])
    batch = list()
    nodes = list()
    out_timesteps = list()
    for timestep in nodes_per_ts.keys():
        print("@@@")
        scene_graph = scene.get_scene_graph(timestep,
                                            env.attention_radius,
                                            hyperparams['edge_addition_filter'],
                                            hyperparams['edge_removal_filter'])
        present_nodes = nodes_per_ts[timestep]
        print("debug1")
        print(nodes_per_ts)
        for node in present_nodes:
            nodes.append(node)
            out_timesteps.append(timestep)
            batch.append(get_node_timestep_data(env, scene, timestep, node, state, pred_state,
                                            edge_types, max_ht, max_ft, hyperparams,
                                            scene_graph=scene_graph, pred_model_type=pred_model_type))

    if len(out_timesteps) == 0:
        return None
    print("debugdebug")
    print(nodes)
    print(out_timesteps)
    return collate(batch), nodes, out_timesteps


def get_timesteps_data_stgcnn(env, scene, t, node_type, state, pred_state,
                       edge_types, min_ht, max_ht, min_ft, max_ft, hyperparams, pred_model_type=2):
    """
    Puts together the inputs for ALL nodes in a given scene and timestep in it.

    :param env: Environment
    :param scene: Scene
    :param t: Timestep in scene
    :param node_type: Node Type of nodes for which the data shall be pre-processed
    :param state: Specification of the node state
    :param pred_state: Specification of the prediction state
    :param edge_types: List of all Edge Types for which neighbors are pre-processed
    :param max_ht: Maximum history timesteps
    :param max_ft: Maximum future timesteps (prediction horizon)
    :param hyperparams: Model hyperparameters
    :return:
    """

    pred_len = 6
    obs_len = 8
    seq_len = 14  # 8+6
    threshold = 0.002
    skip = 1
    min_ped = 1
    max_peds_in_frame = 0
    norm_lap_matr = True

    batch = list
    nodes_list = list()
    out_timesteps = list()

    seq_list = []
    seq_list_rel = []
    loss_mask_list = []
    non_linear_ped = []
    num_peds_in_seq = []
    data = []
    ped_id = []
    present_node_dict = scene.present_nodes(t,
                                           type=node_type,
                                           min_history_timesteps=min_ht,
                                           min_future_timesteps=max_ft,
                                           return_robot=not hyperparams['incl_robot_node'])
    for t, nodes in present_node_dict.items():
        # format: [frame_id, ped_id, x, y]
        for node in nodes:
            out_timesteps.append(t)
            nodes_list.append(node)
            # print(5 * '#')
            delay = 4
            timestep_range_x = np.array([t - obs_len + delay, t + delay])
            timestep_range_y = np.array([t + delay, t + pred_len + delay])

            x = node.get(timestep_range_x, state[node.type])
            y = node.get(timestep_range_y, pred_state[node.type])
            first_history_index = (max_ht - node.history_points_at(t)).clip(0)

            _, std = env.get_standardize_params(state[node.type], node.type)
            std[0:2] = env.attention_radius[(node.type, node.type)]
            rel_state = np.zeros_like(x[0])
            rel_state[0:2] = np.array(x)[-1, 0:2]
            x_st = env.standardize(x, state[node.type], node.type, mean=rel_state, std=std)

            x = x[:, 0:2]
            y = y[:, 0:2]
            x_st = x_st[:, 0:2]

            # x = node.get(t_np, state[node.type])
            # xx, yy = x[0][0], x[0][1]
            if node.id in ped_id:
                node_id = ped_id.index(node.id)
            else:
                ped_id.append(node.id)
                node_id = ped_id.index(node.id)
            # construct path_point
            len_x = x.shape[0]
            for i in range(len_x):
                path_point = [i, node_id, x[i][0], x[i][1]]
                data.append(path_point)
            len_y = y.shape[0]
            for i in range(len_y):
                path_point = [i+len_x, node_id, y[i][0], y[i][1]]
                data.append(path_point)

        # for node in nodes:
        # index += [(scene, t, node)] * \
        #          (scene.frequency_multiplier if scene_freq_mult else 1) * \
        #          (node.frequency_multiplier if node_freq_mult else 1)
    data = np.array(data)
    if len(data) == 0:
        return
    frames = np.unique(data[:, 0]).tolist()
    frame_data = []
    for frame in frames:
        frame_data.append(data[frame == data[:, 0], :])
    num_sequences = int(
        math.ceil((len(frames) - seq_len + 1) / skip))

    for idx in range(0, num_sequences * skip + 1, skip):
        curr_seq_data = np.concatenate(
            frame_data[idx:idx + seq_len], axis=0)
        peds_in_curr_seq = np.unique(curr_seq_data[:, 1])
        # print('*'*30)
        # print('current pedestrian index:',peds_in_curr_seq)
        max_peds_in_frame = max(max_peds_in_frame, len(peds_in_curr_seq))
        curr_seq_rel = np.zeros((len(peds_in_curr_seq), 2,
                                 seq_len))
        curr_seq = np.zeros((len(peds_in_curr_seq), 2, seq_len))
        curr_loss_mask = np.zeros((len(peds_in_curr_seq),
                                   seq_len))
        num_peds_considered = 0
        _non_linear_ped = []
        for _, ped_id in enumerate(peds_in_curr_seq):
            curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] ==
                                         ped_id, :]
            curr_ped_seq = np.around(curr_ped_seq, decimals=4)  # format: [frame_id, ped_id, x, y]
            # pad_front = frames.index(curr_ped_seq[0, 0]) - idx
            # pad_end = frames.index(curr_ped_seq[-1, 0]) - idx + 1
            # print("ped 1:",curr_ped_seq)

            if curr_ped_seq.shape[0] != seq_len:
                continue
            print('ped id:', ped_id)
            curr_frame_id = curr_ped_seq[0, 0]
            # print('frame id:', curr_frame_id)

            # Make coordinates relative
            curr_ped_seq = np.transpose(curr_ped_seq[:, 2:])
            rel_curr_ped_seq = np.zeros(curr_ped_seq.shape)
            rel_curr_ped_seq[:, 1:] = \
                curr_ped_seq[:, 1:] - curr_ped_seq[:, :-1]
            # print("ped 2:",curr_ped_seq)
            _idx = num_peds_considered
            # print("curr_seq:",curr_seq)
            curr_seq[_idx, :, :] = curr_ped_seq
            curr_seq_rel[_idx, :, :] = rel_curr_ped_seq
            # Linear vs Non-Linear Trajectory
            _non_linear_ped.append(
                poly_fit(curr_ped_seq, pred_len, threshold))
            curr_loss_mask[_idx, :] = 1
            num_peds_considered += 1
        if num_peds_considered > min_ped:
            non_linear_ped += _non_linear_ped
            num_peds_in_seq.append(num_peds_considered)
            loss_mask_list.append(curr_loss_mask[:num_peds_considered])
            seq_list.append(curr_seq[:num_peds_considered])
            seq_list_rel.append(curr_seq_rel[:num_peds_considered])

    num_seq = len(seq_list) # 0
    seq_list = np.concatenate(seq_list, axis=0)
    seq_list_rel = np.concatenate(seq_list_rel, axis=0)
    loss_mask_list = np.concatenate(loss_mask_list, axis=0)
    non_linear_ped = np.asarray(non_linear_ped)

    # Convert numpy -> Torch Tensor
    obs_traj = torch.from_numpy(
        seq_list[:, :, :obs_len]).type(torch.float)
    pred_traj = torch.from_numpy(
        seq_list[:, :, obs_len:]).type(torch.float)
    obs_traj_rel = torch.from_numpy(
        seq_list_rel[:, :, :obs_len]).type(torch.float)
    pred_traj_rel = torch.from_numpy(
        seq_list_rel[:, :, obs_len:]).type(torch.float)
    loss_mask = torch.from_numpy(loss_mask_list).type(torch.float)
    non_linear_ped = torch.from_numpy(non_linear_ped).type(torch.float)
    cum_start_idx = [0] + np.cumsum(num_peds_in_seq).tolist()
    seq_start_end = [
        (start, end)
        for start, end in zip(cum_start_idx, cum_start_idx[1:])
    ]
    # Convert to Graphs
    v_obs = []
    A_obs = []
    v_pred = []
    A_pred = []
    print("Processing Data .....")
    for ss in range(len(seq_start_end)):
        start, end = seq_start_end[ss]
        # print(self.obs_traj.shape)  # [65287, 2, 8]
        v_, a_ = seq_to_graph(obs_traj[start:end, :], obs_traj_rel[start:end, :], norm_lap_matr)
        v_obs.append(v_.clone())
        A_obs.append(a_.clone())
        v_, a_ = seq_to_graph(pred_traj[start:end, :], pred_traj_rel[start:end, :], norm_lap_matr)
        v_pred.append(v_.clone())
        A_pred.append(a_.clone())


    index = 0
    start, end = seq_start_end[0]

    out = [
        obs_traj[start:end, :], pred_traj[start:end, :],
        obs_traj_rel[start:end, :], pred_traj_rel[start:end, :],
        non_linear_ped[start:end], loss_mask[start:end, :],
        v_obs[index], A_obs[index],
        v_pred[index], A_pred[index]
    ]

    # print("debugdebug")
    # print(len(nodes_list))
    # print(len(out_timesteps))
    return out, nodes_list, out_timesteps
