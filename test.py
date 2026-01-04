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


def cleanup():
    import torch.distributed as dist

    dist.destroy_process_group()


def now_str():
    from datetime import datetime

    return datetime.now().isoformat(timespec='seconds').replace(':', '_')


@profile
def run(args):
    rank, world_size = setup(debug=False)

    try:
        import pickle
        import json
        from copy import deepcopy

        import torch

        from torch.nn.parallel import DistributedDataParallel as DDP
        from torch.optim.swa_utils import update_bn

        from protoflow.proto import ProtoFlowGMM
        from protoflow.evaluation import test
        from protoflow.utils import dict_to_namespace
        from protoflow.utils import convert_legacy_config
        from protoflow.utils_heavy import make_dataloader

        torch.backends.cudnn.benchmark = True

        run_folder = osp.dirname(args.resume)
        if rank == 0:
            print(f'Run folder: {run_folder}')

        config_path = osp.join(run_folder, 'config.json')
        if osp.exists(config_path):
            with open(config_path, 'r') as fp:
                config = json.load(fp)
            config = convert_legacy_config(config)
            # LDL任务：从配置文件获取npy文件路径
            train_feature_path = config.get('train_feature', args.train_feature)
            train_label_path = config.get('train_label', args.train_label)
            test_feature_path = config.get('test_feature', args.test_feature)
            test_label_path = config.get('test_label', args.test_label)
            # LDL任务：获取模型相关参数
            protos_per_class = config.get('protos_per_class', 10)
            gaussian_approach = config.get('gaussian_approach', 'GaussianMixture')
            likelihood_approach = config.get('likelihood_approach', 'total')
        else:
            if rank == 0:
                print(f'WARNING: no config file found at {config_path}')
            # LDL任务：从命令行参数获取npy文件路径
            train_feature_path = args.train_feature
            train_label_path = args.train_label
            test_feature_path = args.test_feature
            test_label_path = args.test_label
            # LDL任务：默认模型参数
            protos_per_class = 10
            gaussian_approach = 'GaussianMixture'
            likelihood_approach = 'total'

        # LDL任务：加载npy数据，获取标签分布的类别数
        train_labels = np.load(train_label_path)
        n_classes = train_labels.shape[1]  # 标签分布的维度即为类别数
        
        # LDL任务：加载训练数据特征，获取特征维度
        train_features = np.load(train_feature_path)
        feature_dim = train_features.shape[1]  # 特征维度

        # LDL任务：初始化模型，无需flow模型，直接使用ProtoFlowGMM处理特征
        # 需要核实：ProtoFlowGMM是否可以直接处理特征输入，还是必须通过flow模型
        model = ProtoFlowGMM(
            model=None,  # LDL任务：直接处理特征，无需flow模型
            n_classes=n_classes,
            features_shape=(feature_dim,),  # LDL任务：特征形状为(feature_dim,)
            protos_per_class=protos_per_class,
            gaussian_approach=gaussian_approach,
            likelihood_approach=args.likelihood_approach or likelihood_approach,
        )
        model.to(rank)
        model.eval()

        model = DDP(model, device_ids=[rank])

        if rank == 0:
            print(f'Loading checkpoint {args.resume}')
        checkpoint = torch.load(args.resume, map_location=f'cuda:{rank}')
        model.module.load_state_dict(checkpoint['model'])

        if args.var_temp is not None:
            for gmm in model.module.gmms:
                gmm.var.data = gmm.var * args.var_temp

        # LDL任务：使用NpyDataset加载数据，无需数据转换
        dl_train = None
        if not args.test_only:
            ds_train = NpyDataset(train_feature_path, train_label_path)
            dl_train = make_dataloader(ds_train, rank, world_size, args.batch_size)
        ds_test = NpyDataset(test_feature_path, test_label_path)
        dl_test = make_dataloader(ds_test, rank, world_size, args.batch_size)

        to_eval = [('raw', model)]

        if config.get('use_ema', False):
            model_ema = deepcopy(model)
            model_ema.module.load_state_dict(checkpoint['model_ema'])
            if not (checkpoint['model_ema_updated_bn'] or args.no_ema_stats):
                if rank == 0:
                    print('Updating batch norm stats for the EMA model')

                # LDL任务：使用原始训练数据更新EMA模型的batch norm统计信息
                ds_train_plain = NpyDataset(train_feature_path, train_label_path)
                dl_train_plain = make_dataloader(
                    ds_train_plain, rank, world_size, args.batch_size,
                    persistent_workers=False
                )

                update_bn(dl_train_plain, model_ema, device=rank)
                del dl_train_plain

            to_eval.append(('EMA', model_ema))

        for model_name, the_model in to_eval:
            train_scores = None
            if not args.test_only:
                if rank == 0:
                    print(f'Evaluating {model_name} model on train set...')
                train_scores = test(
                    rank=rank,
                    model=the_model,
                    dl=dl_train,
                    n_classes=n_classes,
                    num_samples=args.num_samples,
                    calibration_metrics=True,
                    args=args,
                )
                if rank == 0:
                    print(f'EVALUATION STATS (train, {model_name}):')
                    for name, score in train_scores.items():
                        print(f'  {name}: {score.item():.3f}')

            if rank == 0:
                print(f'Evaluating {model_name} model on test set...')
            test_scores = test(
                rank=rank,
                model=the_model,
                dl=dl_test,
                n_classes=n_classes,
                num_samples=args.num_samples,
                calibration_metrics=True,
                args=args,
            )
            if rank == 0:
                print(f'EVALUATION STATS (test, {model_name}):')
                for name, score in test_scores.items():
                    print(f'  {name}: {score.item():.3f}')

                result_dir = osp.join(run_folder, f'scores_{model_name}')
                os.makedirs(result_dir, exist_ok=True)
                write_path = osp.join(result_dir, f'scores_{now_str()}.json')
                print(f'Write results to {write_path}')
                score_data = {
                    'resume': args.resume,
                    'num_samples': args.num_samples,
                    'var_temp': args.var_temp,
                    'likelihood_approach': args.likelihood_approach,
                    'scores_test': {
                        k: v.item() for k, v in test_scores.items()},
                    # LDL任务：添加npy文件路径信息
                    'train_feature': train_feature_path,
                    'train_label': train_label_path,
                    'test_feature': test_feature_path,
                    'test_label': test_label_path,
                }
                if not args.test_only:
                    score_data['scores_train'] = {
                        k: v.item() for k, v in train_scores.items()}
                with open(write_path, 'w') as fp:
                    json.dump(score_data, fp, indent=2)
    finally:
        cleanup()


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Test a ProtoFlow model for LDL (Label Distribution Learning)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument('--resume', type=str, required=True,
                        help='Path to resume checkpoint')
    
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
    parser.add_argument('--batch_size', '-b', type=int, default=2048)
    parser.add_argument('--batch_steps', '-s', type=int, default=1)
    parser.add_argument('--test_only', action='store_true',
                        help='Only evaluation on the test split')
    parser.add_argument('--no_ema_stats', action='store_true',
                        help='If applicable, do not compute EMA statistics')
    parser.add_argument('--num_samples', '-n', type=int, default=10,
                        help='Number of monte carlo samples')
    
    # LDL任务：移除图像相关参数
    # parser.add_argument('--ten_crop', action='store_true',
    #                     help='Evaluate on 10 crops of the same image')
    # parser.add_argument('--tta', action='store_true',
    #                     help='Use test time augmentation')
    # parser.add_argument('--tta_num', type=int, default=5,
    #                     help='Number of test time augmentations to use')
    
    parser.add_argument('--var_temp', '-T', type=float, default=None,
                        help='Variance temperature')
    parser.add_argument('--likelihood_approach', default=None,
                        choices=('total', 'max'),
                        help='GMM likelihood approach per class')

    args = parser.parse_args()
    print(args)

    run(args)


if __name__ == '__main__':
    main()
