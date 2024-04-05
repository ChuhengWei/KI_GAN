import argparse
import os
import random
import torch
import matplotlib.lines as mlines

from attrdict import AttrDict

from kigan.data.loader import data_loader
from kigan.models import TrajectoryGenerator
from kigan.utils import relative_to_abs, get_dset_path

import numpy as np
import osmnx as ox
import matplotlib.pyplot as plt


def get_generator(checkpoint):
    args = AttrDict(checkpoint['args'])
    generator = TrajectoryGenerator(
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        embedding_dim=args.embedding_dim,
        encoder_h_dim=args.encoder_h_dim_g,
        decoder_h_dim=args.decoder_h_dim_g,
        mlp_dim=args.mlp_dim,
        num_layers=args.num_layers,
        noise_dim=args.noise_dim,
        noise_type=args.noise_type,
        noise_mix_type=args.noise_mix_type,
        pooling_type=args.pooling_type,
        pool_every_timestep=args.pool_every_timestep,
        dropout=args.dropout,
        bottleneck_dim=args.bottleneck_dim,
        neighborhood_size=args.neighborhood_size,
        grid_size=args.grid_size,
        batch_norm=args.batch_norm)
    generator.load_state_dict(checkpoint['g_state'])
    generator.cuda()
    generator.train()
    return generator




def convert_to_latlon(traj, conversion_factor=111000):
    lat = traj[:, :, 0] / conversion_factor
    lon = traj[:, :, 1] / conversion_factor
    return np.stack([lat, lon], axis=2)





def plot_trajectories_on_map(real_traj, pred_traj, seq_start_end, plot_number):
    G = ox.graph_from_xml("map.osm")
    bgcolor = 'white'
    edge_color = 'black'
    fig, ax = ox.plot_graph(G, bgcolor=bgcolor, edge_color=edge_color, show=False, close=False)

    # 定义颜色映射
    cmap = plt.get_cmap('Reds')
    cmap2 = plt.get_cmap('Greens')

    for (start, end) in seq_start_end:
        end = min(end, real_traj.shape[0])
        for i in range(start, end):
            num_points = real_traj.shape[0]
            for point in range(num_points - 1):
                color = cmap(float(point) / num_points)
                plt.plot(real_traj[point:point + 2, i, 0], real_traj[point:point + 2, i, 1], color=color, marker='o')

            num_points_pred = pred_traj[13:31].shape[0]
            for point in range(num_points_pred - 1):
                color = cmap2(float(point) / num_points_pred)
                plt.plot(pred_traj[13:31][point:point + 2, i, 0], pred_traj[13:31][point:point + 2, i, 1], color=color,marker='o')

    red_line = mlines.Line2D([], [], color='red', marker='_', markersize=15, label='Ground Truth')
    green_line = mlines.Line2D([], [], color='green', marker='_', markersize=15, label='Predicted')

    ax.legend(handles=[red_line, green_line])


    if not os.path.exists('Results1212'):
        os.makedirs('Results1212')

    plt.savefig(f'Results1212/trajectory_{plot_number}.png')
    plt.close(fig)

def detect(args, loader, generator, num_samples):

    num_batches = len(loader)
    if num_samples > num_batches:
        num_samples = num_batches
    selected_batches = np.random.choice(num_batches, num_samples, replace=False)




    counter = 0
    plot_number = 1
    for batch in loader:
        if counter in selected_batches:
            batch = [tensor.cuda() for tensor in batch]
            (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped,
             loss_mask, seq_start_end, vx, vy, ax, ay, agent_type, size, traffic_state) = batch

            pred_traj_fake_rel = generator(
                obs_traj, obs_traj_rel, seq_start_end, vx, vy, ax, ay, agent_type, size, traffic_state
            )
            pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])

            real_traj = torch.cat([obs_traj, pred_traj_gt], dim=0).detach().cpu().numpy()

            pred_traj = torch.cat([obs_traj, pred_traj_fake], dim=0).detach().cpu().numpy()

            real_traj_latlon = convert_to_latlon(real_traj)
            pred_traj_latlon = convert_to_latlon(pred_traj)

            plot_trajectories_on_map(real_traj_latlon, pred_traj_latlon, seq_start_end, plot_number)
            plot_number += 1

        counter += 1
        if counter >= num_batches:
            break

def main(args):
    seed = 1234
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    checkpoint = torch.load(args.model_path)
    generator = get_generator(checkpoint)
    _args = AttrDict(checkpoint['args'])
    path = get_dset_path(_args.dataset_name, 'train')
    _, loader = data_loader(_args, path)
    detect(_args, loader, generator, args.num_samples)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='checkpoint_with_model_.pt')
    parser.add_argument('--num_samples', default=200, type=int)
    args = parser.parse_args()
    main(args)
