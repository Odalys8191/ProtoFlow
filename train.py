"""
GNU GPL v2.0
Copyright (c) 2024 Zachariah Carmichael, Timothy Redgrave, Daniel Gonzalez Cedre
ProtoFlow Project

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License along
with this program; if not, write to the Free Software Foundation, Inc.,
51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
"""

import os
import os.path as osp
from contextlib import nullcontext
import numpy as np
import torch
from torch.utils.data import Dataset

from protoflow.utils import profile

class NpyDataset(Dataset):
    """
    加载npy格式的特征和标签分布的数据集类
    适配LDL（标签分布学习）任务
    """
    def __init__(self, feature_path, label_path):
        """
        初始化NpyDataset
        Args:
            feature_path: 特征文件路径（.npy格式）
            label_path: 标签分布文件路径（.npy格式）
        """
        # 加载特征和标签分布
        self.features = np.load(feature_path)
        self.labels = np.load(label_path)
        
        # 确保特征和标签数量一致
        assert len(self.features) == len(self.labels), "特征和标签数量不一致"
        
        # 将数据转换为torch张量
        self.features = torch.tensor(self.features, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.float32)
    
    def __len__(self):
        """
        返回数据集大小
        """
        return len(self.features)
    
    def __getitem__(self, idx):
        """
        获取指定索引的样本
        Args:
            idx: 样本索引
        Returns:
            feature: 特征张量
            label: 标签分布张量
        """
        return self.features[idx], self.labels[idx]

LOG_DIR = os.getenv('LOG_DIR', 'logs')


def setup(debug):
    import torch

    if debug:
        rank = local_rank = 0
        world_size = 1
        torch.cuda.set_device(local_rank)
    else:
        import torch.distributed as dist

        try:
            rank = int(os.environ['RANK'])
            local_rank = int(os.environ['LOCAL_RANK'])
            world_size = int(os.environ['WORLD_SIZE'])

            torch.cuda.set_device(local_rank)
            dist.init_process_group('nccl')
        except KeyError:
            rank = local_rank = 0
            world_size = 1

            torch.cuda.set_device(local_rank)
            dist.init_process_group(
                backend='nccl',
                init_method='tcp://127.0.0.1:12584',
                rank=rank,
                world_size=world_size,
            )

    return rank, world_size


def cleanup(debug):
    from protoflow.datasets import cleanup_shm_data

    if not debug:
        import torch.distributed as dist

        dist.destroy_process_group()

    cleanup_shm_data()


@profile
def run(args):
    rank, world_size = setup(args.debug)
    args.world_size = world_size
    try:
            import pickle
            import json

            import git

            import torch
            from torch.nn.parallel import DistributedDataParallel as DDP
            from torch.utils.data import random_split
            from torch.utils.data import Subset

            from experiments.image.model.model_flow import get_model
            from experiments.image.utils import set_seeds
            from protoflow.proto import ProtoFlowGMM
            from protoflow.training import train_loop
            from protoflow.training import init_gmms
            from protoflow.evaluation import test
            from protoflow.utils_heavy import make_dataloader

            torch.backends.cudnn.benchmark = True
            set_seeds(seed=0xFACE)

            try:
                repo = git.Repo(search_parent_directories=True)
                sha = repo.head.object.hexsha
            except git.InvalidGitRepositoryError:
                sha = None
            args.git_commit_hash = sha

            if args.detect_autograd_anomaly:
                torch.autograd.set_detect_anomaly(True)

            # LDL任务：使用npy文件加载数据，无需图像相关参数
            name = f'bs{args.batch_size}_ldl'
            if args.extra is not None:
                run_folder = osp.join(LOG_DIR, f'ldl_{args.extra}', name)
            else:
                run_folder = osp.join(LOG_DIR, 'ldl', name)

            # LDL任务：从命令行参数获取npy文件路径
            train_feature_path = args.train_feature
            train_label_path = args.train_label
            test_feature_path = args.test_feature
            test_label_path = args.test_label

            if rank == 0:
                os.makedirs(run_folder, exist_ok=True)
                print(f'Run folder: {run_folder}')

                config_path = osp.join(run_folder, 'config.json')
                if not osp.exists(config_path):
                    if args.resume:
                        print(
                            f'WARNING: no config file found at {config_path} even '
                            f'though we are resuming training! Creating one now '
                            f'based on your arguments.')
                    with open(config_path, 'w') as fp:
                        json.dump({
                            'batch_size': args.batch_size,
                            'world_size': args.world_size,
                            'num_epochs': args.num_epochs,
                            'lr': args.lr,
                            'gmm_lr': args.gmm_lr,
                            'use_ema': args.use_ema,
                            'warmup_epochs': args.warmup_epochs,
                            'trainable': args.trainable,
                            'gmm_em': args.gmm_em,
                            'init_gmm': args.init_gmm,
                            'clip_grad_norm': args.clip_grad_norm,
                            'weight_decay': args.weight_decay,
                            'protos_per_class': args.protos_per_class,
                            'gaussian_approach': args.gaussian_approach,
                            'likelihood_approach': args.likelihood_approach,
                            'proto_dropout_prob': args.proto_dropout_prob,
                            'z_dropout_prob': args.z_dropout_prob,
                            'mu_loss': args.mu_loss,
                            'proto_loss': args.proto_loss,
                            'git_commit_hash': args.git_commit_hash,
                            # LDL任务新增参数
                            'train_feature': args.train_feature,
                            'train_label': args.train_label,
                            'test_feature': args.test_feature,
                            'test_label': args.test_label,
                        }, fp, indent=2)

            # LDL任务：加载npy数据，获取标签分布的类别数
            train_labels = np.load(train_label_path)
            n_classes = train_labels.shape[1]  # 标签分布的维度即为类别数
            
            # LDL任务：加载训练数据特征，获取特征维度
            train_features = np.load(train_feature_path)
            feature_dim = train_features.shape[1]  # 特征维度
            
            # LDL任务：初始化模型，无需flow模型，直接使用ProtoFlowGMM处理特征
            # 注意：这里需要确认ProtoFlowGMM是否可以直接处理特征输入
            # 如果需要，可能需要修改ProtoFlowGMM或添加适配层
            # 需要核实：ProtoFlowGMM是否可以直接处理特征输入，还是必须通过flow模型
            # 目前假设ProtoFlowGMM可以直接处理特征输入
            model = ProtoFlowGMM(
                model=None,  # LDL任务：直接处理特征，无需flow模型
                n_classes=n_classes,
                features_shape=(feature_dim,),  # LDL任务：特征形状为(feature_dim,)
                protos_per_class=args.protos_per_class,
                gaussian_approach=args.gaussian_approach,
                likelihood_approach=args.likelihood_approach,
                proto_dropout_prob=args.proto_dropout_prob,
                z_dropout_prob=args.z_dropout_prob,
            )
            model.to(rank)

            if not args.debug:
                model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

                model = DDP(model, device_ids=[rank],
                            find_unused_parameters=(args.trainable == 'gmm'),
                            broadcast_buffers=False)


            else:
                model.module = model

            assert args.batch_size % args.batch_steps == 0, 'batch step must divide batch size evenly'

            # LDL任务：使用NpyDataset加载训练数据
            ds_train = NpyDataset(train_feature_path, train_label_path)
            val_len = round(len(ds_train) * args.val_size)
            if val_len == 0 and args.val_size != 0 and len(ds_train) > 1:
                val_len = 1
            if val_len != 0:
                train_len = len(ds_train) - val_len
                g = torch.Generator().manual_seed(42)
                ds_train, ds_val = random_split(ds_train, [train_len, val_len],
                                                generator=g)
            else:
                ds_val = None

            if args.init_gmm:
                if args.resume:
                    print(
                        f'Ignoring option --init_gmm {args.init_gmm} because we are '
                        f'resuming training')
                else:
                    print('Load train data for GMM initialization')
                    # LDL任务：无需数据增强，直接使用原始数据
                    ds_train_init = NpyDataset(train_feature_path, train_label_path)
                    if val_len != 0:
                        ds_train_init = Subset(ds_train_init, ds_train.indices)
                    dl_train_init = make_dataloader(
                        ds_train_init, rank, world_size, args.batch_size,
                        drop_last=True, shuffle=False, persistent_workers=False,
                        num_workers=args.num_workers,
                        prefetch_cuda=args.prefetch_cuda,
                    )
                    init_gmms(rank, world_size, model, dl_train_init, args,
                              method=args.init_gmm)

                    del dl_train_init, ds_train_init

            dl_train = make_dataloader(ds_train, rank, world_size, args.batch_size,
                                       drop_last=True, shuffle=True,
                                       num_workers=args.num_workers,
                                       prefetch_cuda=args.prefetch_cuda)
            if ds_val is None:
                dl_val = None
            else:
                dl_val = make_dataloader(ds_val, rank, world_size, args.batch_size,
                                         drop_last=False, shuffle=False,
                                         num_workers=args.num_workers,
                                         prefetch_cuda=args.prefetch_cuda)

            with torch.no_grad() if args.gmm_em else nullcontext():
                train_loop(
                    rank=rank,
                    world_size=world_size,
                    model=model,
                    run_folder=run_folder,
                    dl_train=dl_train,
                    dl_val=dl_val,
                    n_classes=n_classes,
                    args=args,
                )

            if rank == 0:
                print('Test accuracy time')
            # LDL任务：使用NpyDataset加载测试数据
            ds_test = NpyDataset(test_feature_path, test_label_path)
            dl_test = make_dataloader(ds_test, rank, world_size, args.batch_size,
                                      num_workers=args.num_workers,
                                      prefetch_cuda=args.prefetch_cuda)
            test_scores = test(
                rank=rank,
                model=model,
                dl=dl_test,
                n_classes=n_classes,
                args=args,
            )
            if rank == 0:
                print('EVALUATION STATS:')
                for name, score in test_scores.items():
                    print(f'  {name}: {score.item():.3f}')
    finally:
        cleanup(args.debug)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Train a ProtoFlow model for LDL (Label Distribution Learning)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # LDL任务：新增npy文件路径参数
    parser.add_argument('--train_feature', type=str, required=True,
                        help='Training feature file path (.npy format)')
    parser.add_argument('--train_label', type=str, required=True,
                        help='Training label distribution file path (.npy format)')
    parser.add_argument('--test_feature', type=str, required=True,
                        help='Test feature file path (.npy format)')
    parser.add_argument('--test_label', type=str, required=True,
                        help='Test label distribution file path (.npy format)')

    # 保留原有必要参数
    parser.add_argument('--batch_size', '-b', type=int, default=32)
    parser.add_argument('--batch_steps', '-s', type=int, default=1)
    parser.add_argument('--num_epochs', '-e', type=int, default=50)
    parser.add_argument('--warmup_epochs', '-w', type=float, default=5)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--gmm_lr', type=float, default=1e-3)
    parser.add_argument('--extra', type=str, default=None,
                        help='To append to the run folder name')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to resume checkpoint')
    parser.add_argument('--no_restore_optimizer', action='store_true',
                        help='Do not restore the optimizer and scheduler')
    parser.add_argument('--use_ema', action='store_true',
                        help='Use EMA (exponential moving average) of model weights')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='Optimizer weight decay')
    # LDL任务：修改trainable选项，移除flow相关选项
    parser.add_argument('--trainable', default='gmm',
                        choices=('gmm', 'all', 'all_means'),
                        help='Which parts of the model to train')
    parser.add_argument('--save_k_best', default=2, type=int)
    parser.add_argument('--val_size', default=0.1, type=float)
    parser.add_argument('--clip_grad_norm', default=100.0, type=float)
    parser.add_argument('--debug', action='store_true',
                        help='Debugging')
    parser.add_argument('--detect_autograd_anomaly', action='store_true',
                        help='Set autograd detect anomaly to true')

    # LDL任务：移除flow_ckpt相关参数，无需flow模型
    # parser.add_argument('--flow_ckpt', type=str, required=True)
    # parser.add_argument('--no_restore_flow_ckpt', action='store_true')
    
    # 保留原有GMM相关参数
    parser.add_argument('--gmm_em', action='store_true',
                        help='Train GMMs using EM only')
    parser.add_argument('--protos_per_class', type=int, default=10,
                        help='Number of prototypical distributions per class')
    parser.add_argument('--init_gmm', default=None,
                        choices=('kmeans',),
                        help='Init GMM means using this method')
    parser.add_argument('--gaussian_approach', default='GaussianMixture',
                        choices=('GaussianMixture', 'SSLGaussMixture',
                                 'GaussianMixtureConv2d'),
                        help='GMM approach')
    parser.add_argument('--likelihood_approach', default='total',
                        choices=('total', 'max'),
                        help='GMM likelihood approach per class')
    parser.add_argument('--proto_dropout_prob', default=None, type=float,
                        help='GMM prototype dropout prob')
    parser.add_argument('--z_dropout_prob', default=None, type=float,
                        help='Embedding dropout prob')
    parser.add_argument('--mu_loss', action='store_true',
                        help='Add cluster loss term for means of GMMs')
    parser.add_argument('--proto_loss', action='store_true',
                        help='Add elbo loss term for means of GMMs')
    # LDL任务：移除图像相关的损失函数
    # parser.add_argument('--elbo_loss', action='store_true',
    #                     help='Add elbo loss term for z...')
    # parser.add_argument('--elbo_loss2', action='store_true',
    #                     help='Add different elbo loss term for z...')
    # parser.add_argument('--consistency_loss', action='store_true',
    #                     help='Add consistency loss term')
    # parser.add_argument('--consistency_rampup', type=int, default=50,
    #                     help='Consistency loss rampup epochs')
    parser.add_argument('--diversity_loss', action='store_true',
                        help='Add intra-class prototype diversity loss term')

    # 保留原有DataLoader相关参数
    parser.add_argument('--num_workers', default=None, type=int,
                        help='Number of workers per DataLoader')
    parser.add_argument('--prefetch_cuda', action='store_true',
                        help='Prefetch data to CUDA devices')

    args = parser.parse_args()
    print(args)

    run(args)


if __name__ == '__main__':
    main()
