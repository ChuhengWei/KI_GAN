import logging
import os
import math

import numpy as np

import torch
from torch.utils.data import Dataset

_obs_len = 12
_pred_len = 18
logger = logging.getLogger(__name__)



def seq_collate(data):
    (obs_seq_list, pred_seq_list, obs_seq_rel_list, pred_seq_rel_list,
     non_linear_ped_list, loss_mask_list,
     # 【添加】
     vx_list,vy_list,ax_list,ay_list,
     agent_type_list, size_list,traffic_state_list) = zip(*data)



    _len = [len(seq) for seq in obs_seq_list]
    cum_start_idx = [0] + np.cumsum(_len).tolist()
    seq_start_end = [[start, end]
                     for start, end in zip(cum_start_idx, cum_start_idx[1:])]


    obs_traj = torch.cat(obs_seq_list, dim=0).permute(2, 0, 1)
    pred_traj = torch.cat(pred_seq_list, dim=0).permute(2, 0, 1)
    obs_traj_rel = torch.cat(obs_seq_rel_list, dim=0).permute(2, 0, 1)
    pred_traj_rel = torch.cat(pred_seq_rel_list, dim=0).permute(2, 0, 1)

    # 【添加】
    vx = torch.cat(vx_list, dim=0).permute(2, 0, 1)
    vy = torch.cat(vy_list, dim=0).permute(2, 0, 1)
    ax = torch.cat(ax_list, dim=0).permute(2, 0, 1)
    ay = torch.cat(ay_list, dim=0).permute(2, 0, 1)
    agent_type = torch.cat(agent_type_list, dim=0).permute(2, 0, 1)
    size = torch.cat(size_list, dim=0).permute(2, 0, 1)
    traffic_state = torch.cat(traffic_state_list, dim=0).permute(2, 0, 1)



    non_linear_ped = torch.cat(non_linear_ped_list)
    loss_mask = torch.cat(loss_mask_list, dim=0)
    seq_start_end = torch.LongTensor(seq_start_end)
    out = [
        obs_traj, pred_traj, obs_traj_rel, pred_traj_rel, non_linear_ped,
        loss_mask, seq_start_end,
        vx,vy,ax,ay,agent_type,size,traffic_state
    ]

    return tuple(out)


def read_file(_path, delim='\t'):
    data = []
    if delim == 'tab':
        delim = '\t'
    elif delim == 'space':
        delim = ' '
    with open(_path, 'r') as f:

        for line in f:
            line = line.strip().split(delim)
            # 【修改】读取时问题
            line = line[0].split(',')  # 从列表中提取字符串并分割
            # print(line)  # 在转换之前打印出line的内容
            line = [float(i) for i in line]
            data.append(line)
    return np.asarray(data)


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


class TrajectoryDataset(Dataset):
    """Dataloder for the Trajectory datasets"""
    def __init__(

        self, data_dir, obs_len= _obs_len , pred_len= _pred_len, skip=15, threshold=30,
        min_ped=1, delim='\t'
    ):
        """
        Args:
        - data_dir: Directory containing dataset files in the format
        <frame_id> <ped_id> <x> <y>
        - obs_len: Number of time-steps in input trajectories
        - pred_len: Number of time-steps in output trajectories
        - skip: Number of frames to skip while making the dataset
        - threshold: Minimum error to be considered for non linear traj
        when using a linear predictor，调整线性和非线性的判定条件
        - min_ped: Minimum number of pedestrians that should be in a seqeunce。序列中至少包括多少个行人
        - delim: Delimiter in the dataset files
        """
        super(TrajectoryDataset, self).__init__()

        self.data_dir = data_dir
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.skip = skip
        self.seq_len = self.obs_len + self.pred_len
        self.delim = delim

        vx_list = []
        vy_list = []
        ax_list = []
        ay_list = []
        agent_type_list = []
        size_list = []
        traffic_state_list = []

        all_files = os.listdir(self.data_dir)
        all_files = [os.path.join(self.data_dir, _path) for _path in all_files]
        num_vehs_in_seq = []
        seq_list = []
        seq_list_rel = []
        loss_mask_list = []
        non_linear_ped = []
        for path in all_files:
            data = read_file(path, delim)
            #获取帧ID，并只保留一个，确保帧ID唯一，每个帧ID在一行
            frames = np.unique(data[:, 0]).tolist()


            frame_data = []
            for frame in frames:
                frame_data.append(data[frame == data[:, 0], :])



            num_sequences = int(
                math.ceil((len(frames) - self.seq_len + 1) / skip))


            for idx in range(0, num_sequences * self.skip + 1, skip):
                curr_seq_data = np.concatenate(
                    frame_data[idx:idx + self.seq_len], axis=0)


                vehs_in_curr_seq = np.unique(curr_seq_data[:, 1])

                curr_seq_rel = np.zeros((len(vehs_in_curr_seq), 2, self.seq_len))
                curr_seq = np.zeros((len(vehs_in_curr_seq), 2, self.seq_len))
                # 【添加】
                curr_vx =np.zeros((len(vehs_in_curr_seq), 1, self.seq_len))
                curr_vy =np.zeros((len(vehs_in_curr_seq), 1, self.seq_len))
                curr_ax =np.zeros((len(vehs_in_curr_seq), 1, self.seq_len))
                curr_ay =np.zeros((len(vehs_in_curr_seq), 1, self.seq_len))
                curr_size =np.zeros((len(vehs_in_curr_seq), 2, self.seq_len))
                curr_agent_type =np.zeros((len(vehs_in_curr_seq), 1, self.seq_len))
                curr_traffic_state =np.zeros((len(vehs_in_curr_seq), 1, self.seq_len))


                curr_loss_mask = np.zeros((len(vehs_in_curr_seq),
                                           self.seq_len))
                
                num_vehs_considered = 0
                _non_linear_ped = []

                for _, ped_id in enumerate(vehs_in_curr_seq):
                    curr_ped_seq_ = curr_seq_data[curr_seq_data[:, 1] == ped_id, :]

                    curr_ped_seq_ = np.around(curr_ped_seq_, decimals=4)


                    pad_front = frames.index(curr_ped_seq_[0, 0]) - idx
                    pad_end = frames.index(curr_ped_seq_[-1, 0]) - idx + 1

                    if pad_end - pad_front != self.seq_len:
                        continue


                    curr_ped_seq = np.transpose(curr_ped_seq_[:,2: 4])
                    rel_curr_ped_seq = np.zeros(curr_ped_seq.shape)
                    rel_curr_ped_seq[:, 1:] = curr_ped_seq[:, 1:] - curr_ped_seq[:, :-1]


                    curr_vel_vx = np.transpose(curr_ped_seq_[:,4])
                    curr_vel_vy = np.transpose(curr_ped_seq_[:, 5])
                    curr_vel_ax = np.transpose(curr_ped_seq_[:, 6])
                    curr_vel_ay = np.transpose(curr_ped_seq_[:, 7])
                    curr_vel_size = np.transpose(curr_ped_seq_[:, 8:10])
                    curr_vel_agent_type=np.transpose(curr_ped_seq_[:,10])
                    curr_vel_traffic_state=np.transpose(curr_ped_seq_[:,11])


                    _idx = num_vehs_considered

                    curr_seq[_idx, :, pad_front:pad_end] = curr_ped_seq
                    curr_seq_rel[_idx, :, pad_front:pad_end] = rel_curr_ped_seq
                    curr_vx[_idx, :, pad_front:pad_end]=curr_vel_vx
                    curr_vy[_idx, :, pad_front:pad_end]=curr_vel_vy
                    curr_ax[_idx, :, pad_front:pad_end]=curr_vel_ax
                    curr_ay[_idx, :, pad_front:pad_end]=curr_vel_ay
                    curr_size[_idx, :, pad_front:pad_end]=curr_vel_size
                    curr_agent_type[_idx, :, pad_front:pad_end]=curr_vel_agent_type
                    curr_traffic_state[_idx, :, pad_front:pad_end]=curr_vel_traffic_state



                    _non_linear_ped.append(
                        poly_fit(curr_ped_seq, pred_len, threshold))
                    curr_loss_mask[_idx, pad_front:pad_end] = 1
                    num_vehs_considered += 1

                if num_vehs_considered > min_ped:
                    non_linear_ped += _non_linear_ped
                    num_vehs_in_seq.append(num_vehs_considered)
                    loss_mask_list.append(curr_loss_mask[:num_vehs_considered])
                    seq_list.append(curr_seq[:num_vehs_considered])
                    seq_list_rel.append(curr_seq_rel[:num_vehs_considered])
                    vx_list.append(curr_vx[:num_vehs_considered])
                    vy_list.append(curr_vy[:num_vehs_considered])
                    ax_list.append(curr_ax[:num_vehs_considered])
                    ay_list.append(curr_ay[:num_vehs_considered])
                    agent_type_list.append(curr_agent_type[:num_vehs_considered])
                    size_list.append(curr_size[:num_vehs_considered])
                    traffic_state_list.append(curr_traffic_state[:num_vehs_considered])


        self.num_seq = len(seq_list)
        seq_list = np.concatenate(seq_list, axis=0)
        seq_list_rel = np.concatenate(seq_list_rel, axis=0)
        loss_mask_list = np.concatenate(loss_mask_list, axis=0)
        non_linear_ped = np.asarray(non_linear_ped)
        # 【添加】
        vx_list = np.concatenate(vx_list, axis=0)
        vy_list = np.concatenate(vy_list, axis=0)
        ax_list = np.concatenate(ax_list, axis=0)
        ay_list = np.concatenate(ay_list, axis=0)
        agent_type_list = np.concatenate(agent_type_list, axis=0)
        size_list = np.concatenate(size_list, axis=0)
        traffic_state_list = np.concatenate(traffic_state_list, axis=0)





        self.obs_traj = torch.from_numpy(seq_list[:, :, :self.obs_len]).type(torch.float)
        self.pred_traj = torch.from_numpy(seq_list[:, :, self.obs_len:]).type(torch.float)
        self.obs_traj_rel = torch.from_numpy(seq_list_rel[:, :, :self.obs_len]).type(torch.float)
        self.pred_traj_rel = torch.from_numpy(seq_list_rel[:, :, self.obs_len:]).type(torch.float)

        self.vx=torch.from_numpy(vx_list[:, :, :self.obs_len]).type(torch.float)
        self.vy=torch.from_numpy(vy_list[:, :, :self.obs_len]).type(torch.float)
        self.ax=torch.from_numpy(ax_list[:, :, :self.obs_len]).type(torch.float)
        self.ay=torch.from_numpy(ay_list[:, :, :self.obs_len]).type(torch.float)
        self.size=torch.from_numpy(size_list[:, :, :self.obs_len]).type(torch.float)
        self.agent_type=torch.from_numpy(agent_type_list[:, :, :self.obs_len]).type(torch.float)
        self.traffic_state=torch.from_numpy(traffic_state_list[:, :, :self.obs_len]).type(torch.float)


        self.loss_mask = torch.from_numpy(loss_mask_list).type(torch.float)
        self.non_linear_ped = torch.from_numpy(non_linear_ped).type(torch.float)
        cum_start_idx = [0] + np.cumsum(num_vehs_in_seq).tolist()
        self.seq_start_end = [
            (start, end)
            for start, end in zip(cum_start_idx, cum_start_idx[1:])
        ]

    def __len__(self):
        return self.num_seq

    def __getitem__(self, index):
        start, end = self.seq_start_end[index]
        out = [
            self.obs_traj[start:end, :], self.pred_traj[start:end, :],self.obs_traj_rel[start:end, :], self.pred_traj_rel[start:end, :],
            self.non_linear_ped[start:end], self.loss_mask[start:end, :],
            self.vx[start:end, :],self.vy[start:end, :],self.ax[start:end, :],self.ay[start:end, :],
            self.agent_type[start:end, :],self.size[start:end, :],self.traffic_state[start:end, :]
        ]
        return out
