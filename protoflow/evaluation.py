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
from tqdm.auto import tqdm

import torch
import numpy as np

# LDL任务：导入LDL指标
from ldl_metrics import proj, Cheby, Clark, Canberra, KL_div, Cosine, Intersection


def test(rank, model, dl, n_classes, args, num_samples=1, ten_crop=False,
         calibration_metrics=False, multi_transform_k=None, log_suffix=None):
    was_training = model.training

    # LDL任务：初始化LDL指标存储
    # 由于LDL指标需要在numpy上计算，这里使用列表存储所有预测和真实标签
    all_preds = []
    all_labels = []
    
    model.eval()

    inner_batch_size = args.batch_size // args.batch_steps
    with torch.no_grad():
        desc = 'Evaluating' + (f' {log_suffix}' if log_suffix else '') + '...'
        for batch_idx, (features, labels) in enumerate(tqdm(
                dl, disable=rank != 0, desc=desc)):
            # LDL任务：处理特征输入而非图像输入
            if isinstance(features, (tuple, list)):
                assert len(features) == 2
                features = features[0]
            features = features.to(rank)
            labels = labels.to(rank)

            for i in range(args.batch_steps):
                idxs_step = slice(
                    i * inner_batch_size, (i + 1) * inner_batch_size)
                features_step = features[idxs_step]
                if not len(features_step):
                    break
                labels_step = labels[idxs_step]
                preds_agg = None
                log_prob_agg = None
                for _ in range(num_samples):
                    # LDL任务：直接使用特征输入，无需数据增强和裁剪
                    divisor = num_samples
                    # LDL任务：调用模型进行预测
                    # 需要核实：模型的forward方法是否支持直接处理特征输入
                    # 以及返回值格式是否正确
                    preds, log_prob = model(
                        features_step, flow_grad=False,
                        ret_log_prob=True, ret_z=False,
                    )

                    if preds_agg is None:
                        preds_agg = preds / divisor
                        log_prob_agg = log_prob / divisor
                    else:
                        preds_agg += preds / divisor
                        log_prob_agg += log_prob / divisor

                # LDL任务：将预测和标签转换为numpy数组，存储到列表中
                # 注意：这里假设preds_agg的形状是(batch_size, n_classes)
                # 且表示的是未归一化的预测分数
                all_preds.append(preds_agg.cpu().numpy())
                all_labels.append(labels_step.cpu().numpy())

        # LDL任务：计算LDL指标
        # 1. 合并所有批次的预测和标签
        all_preds = np.concatenate(all_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        
        # 2. 对预测结果进行投影，确保其在概率单纯形上
        # 注意：这里假设preds_agg是未归一化的预测分数
        # 需要核实：preds_agg的具体含义和格式
        # 如果preds_agg已经是概率分布，则无需投影
        all_preds_proj = proj(all_preds)
        
        # 3. 计算LDL指标
        cheby = Cheby(all_labels, all_preds_proj)
        clark = Clark(all_labels, all_preds_proj)
        canberra = Canberra(all_labels, all_preds_proj)
        kl_div = KL_div(all_labels, all_preds_proj)
        cosine = Cosine(all_labels, all_preds_proj)
        intersection = Intersection(all_labels, all_preds_proj)
        
        # 4. 构建分数字典
        scores = {
            'cheby': torch.tensor(cheby),
            'clark': torch.tensor(clark),
            'canberra': torch.tensor(canberra),
            'kl_div': torch.tensor(kl_div),
            'cosine': torch.tensor(cosine),
            'intersection': torch.tensor(intersection),
        }

    if was_training:
        model.train()

    return scores
