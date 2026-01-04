import os
import json
import argparse
import itertools
import numpy as np
import sys
import time

# 导入train模块
import train

# ================= 🔧 配置区域 =================

# SOTA JSON 路径
SOTA_PATH = "../Data/sota.json"
# 结果保存根目录
RESULT_ROOT = "result"

# 指标名称映射：train.py返回的指标名 -> 输出的指标名
METRICS_MAPPING = {
    'cheby': 'Chebyshev',
    'clark': 'Clark',
    'canberra': 'Canberra',
    'kl_div': 'KL Divergence',
    'cosine': 'Cosine',
    'intersection': 'Intersection'
}

# 映射关系：文件夹名 -> SOTA JSON 里的 Key
DATASET_MAPPING = {
    "Gene": "Gene",
    "Movie": "Movie",
    "RAF_ML": "RAF_ML",
    "Ren_Cecps": "Ren_Cecps",
    "SBU_3DFE": "SBU_3DFE",
    "Scene": "Scene",
    
    "Flickr_LDL": "Flickr_LDL",
    "M2B": "M2B",
    "SCUT_FBP": "SCUT_FBP",
    "SCUT_FBP5500": "SCUT_FBP5500",
    "SJAFFE": "SJAFFE",
    "Twitter_LDL": "Twitter_LDL"
}

# Grid Search 搜索空间
SEARCH_SPACE = {
    "lr": [1e-3, 5e-4, 1e-4, 1e-5],
    "batch_size": [128, 256],
    "feature_dim": [64, 128, 256, 512]
}

# 指标名称
METRICS_NAMES = ['Chebyshev', 'Clark', 'Canberra', 'KL Divergence', 'Cosine', 'Intersection']
SOTA_KEYS = ['Cheby', 'Clark', 'Canbe', 'KL', 'Cosine', 'Inter']

# ================= 功能函数 =================

def get_sota_directly(dataset_folder_name):
    """直接读取 JSON 内容获取SOTA指标"""
    if not os.path.exists(SOTA_PATH):
        print(f"❌ Error: {SOTA_PATH} not found.")
        return None

    try:
        with open(SOTA_PATH, 'r', encoding='utf-8') as f:
            full_data = json.load(f)
        
        data_block = full_data.get('data', {})
        # 通过映射表找到 JSON 里的 key
        target_key = DATASET_MAPPING.get(dataset_folder_name, dataset_folder_name)
        
        if target_key not in data_block:
            print(f"⚠️ SOTA data for '{target_key}' not found (Folder: {dataset_folder_name}).")
            return None
        
        vals = []
        for key in SOTA_KEYS:
            vals.append(data_block[target_key][key]['mean'])
        
        print(f"📚 Loaded SOTA for {dataset_folder_name}: {vals}")
        return vals

    except Exception as e:
        print(f"❌ JSON Error: {e}")
        return None


def calc_avg_imp(our_mean, sota_vals):
    """计算平均改进率"""
    if not sota_vals:
        return 0.0
    imps = []
    for i in range(4): # 前4个指标越小越好
        imps.append((sota_vals[i] - our_mean[i]) / (sota_vals[i] + 1e-12))
    for i in range(4, 6): # 后2个指标越大越好
        imps.append((our_mean[i] - sota_vals[i]) / (sota_vals[i] + 1e-12))
    return np.mean(imps)


def get_run_files(dataset_path, run_idx):
    """获取指定run的文件路径"""
    run_dir = os.path.join(dataset_path, f"run_{run_idx}")
    return {
        "train_feature": os.path.join(run_dir, "train_feature.npy"),
        "train_label": os.path.join(run_dir, "train_label.npy"),
        "test_feature": os.path.join(run_dir, "test_feature.npy"),
        "test_label": os.path.join(run_dir, "test_label.npy")
    }

# ================= 主逻辑 =================

def main(dataset, device):
    # 1. 准备目录
    save_dir = os.path.join(RESULT_ROOT, dataset)
    os.makedirs(save_dir, exist_ok=True)
    txt_path = os.path.join(save_dir, "result.txt")
    best_params_path = os.path.join(save_dir, "best_params.json")
    best_model_path = os.path.join(save_dir, "best_model_path.txt")

    print(f"🚀 Processing Dataset: {dataset} on {device}")
    
    # 2. 获取 SOTA
    sota_vals = get_sota_directly(dataset)

    # 3. Grid Search (在 run_0 上进行5折交叉验证)
    print(f"\n{'='*30}\n🔍 Grid Search (run_0 with 5-fold CV)\n{'='*30}")
    
    keys, values = zip(*SEARCH_SPACE.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    best_avg_imp = -float('inf')
    best_params = {}
    best_cv_metrics = None

    for params in combinations:
        print(f"\nTesting Params: {params}")
        
        # 获取run_0的文件路径
        dataset_path = os.path.join("/home/ubuntu/zxj/Data/feature", dataset)
        run_0_files = get_run_files(dataset_path, 0)
        
        # 直接调用train模块的run函数，使用create_train_args创建参数
        train_args = train.create_train_args(
            train_feature=run_0_files["train_feature"],
            train_label=run_0_files["train_label"],
            test_feature=run_0_files["test_feature"],
            test_label=run_0_files["test_label"],
            device=device,
            num_epochs=200,
            num_workers=0,
            cv_folds=5,  # 五折交叉验证
            extra=f"grid_search_{dataset}",
            **params  # 传递所有超参数
        )
        
        # 调用train.run函数获取交叉验证结果
        cv_results = train.run(train_args)
        
        if cv_results:
            # 将cv_results转换为与METRICS_NAMES对应的列表
            metrics = [cv_results['cheby'], cv_results['clark'], cv_results['canberra'], 
                      cv_results['kl_div'], cv_results['cosine'], cv_results['intersection']]
            avg_imp = calc_avg_imp(metrics, sota_vals)
            print(f"👉 Result Metrics: {metrics}")
            print(f"👉 AvgImp: {avg_imp:.2%}")
            
            if avg_imp > best_avg_imp:
                best_avg_imp = avg_imp
                best_params = params
                best_cv_metrics = metrics
                print("⭐ Current Best!")
        else:
            print("❌ Failed to get cv results")

    print(f"\n✅ Best Params Found: {best_params} (AvgImp: {best_avg_imp:.2%})")
    
    # 保存最优参数
    with open(best_params_path, 'w', encoding='utf-8') as f:
        json.dump(best_params, f, indent=2)
    print(f"💾 Saved best_params.json to {best_params_path}")

    # 4. 跑所有 run_0 至 run_9
    print(f"\n{'='*30}\n🏃 Running 10 runs (run_0 to run_9)\n{'='*30}")
    
    all_run_metrics = []
    best_model_paths = []

    for run_idx in range(10):
        print(f"\n>>> Run {run_idx}/9")
        
        # 获取当前run的文件路径
        run_files = get_run_files(dataset_path, run_idx)
        
        # 创建训练参数
        train_args = train.create_train_args(
            train_feature=run_files["train_feature"],
            train_label=run_files["train_label"],
            test_feature=run_files["test_feature"],
            test_label=run_files["test_label"],
            device=device,
            batch_size=best_params["batch_size"],
            lr=best_params["lr"],
            num_epochs=200,
            num_workers=0,
            cv_folds=1,  # 非交叉验证模式
            extra=f"{dataset}_run{run_idx}",
            **best_params  # 传递其他超参数
        )
        
        print(f"\n🏋️ Training on run {run_idx}...")
        # 调用train.run函数获取测试结果
        test_results = train.run(train_args)
        
        if test_results:
            # 将test_results转换为与METRICS_NAMES对应的列表
            metrics = [test_results['cheby'], test_results['clark'], test_results['canberra'], 
                      test_results['kl_div'], test_results['cosine'], test_results['intersection']]
            all_run_metrics.append(metrics)
            print(f"✅ Run {run_idx} Metrics: {metrics}")
            
            # 保存当前run的最佳模型路径
            # 注意：这里假设train.py会在logs目录下生成模型文件
            log_dir = os.path.join("logs", f"ldl_{dataset}_run{run_idx}")
            if os.path.exists(log_dir):
                for root, dirs, files in os.walk(log_dir):
                    for file in files:
                        if file.endswith(".pt") and "best" in file:
                            model_path = os.path.abspath(os.path.join(root, file))
                            best_model_paths.append(model_path)
                            print(f"📦 Saved model checkpoint: {model_path}")
                            break
        else:
            print(f"❌ Failed to get metrics for run {run_idx}")

    # 5. 存档和结果输出
    if not all_run_metrics:
        print("❌ No results.")
        return

    all_run_metrics = np.array(all_run_metrics)
    means = np.mean(all_run_metrics, axis=0)
    stds = np.std(all_run_metrics, axis=0)
    overall_avg_imp = calc_avg_imp(means, sota_vals)

    lines = []
    lines.append(f"Dataset: {dataset}")
    lines.append(f"Best Params: {best_params}")
    lines.append(f"Best Model Checkpoints:")
    for i, path in enumerate(best_model_paths):
        lines.append(f"  Run {i}: {path}")
    lines.append("-" * 85)
    lines.append(f"{'Metric':<15} | {'Mean ± Std':<25} | {'SOTA':<10}")
    lines.append("-" * 85)
    
    for i, name in enumerate(METRICS_NAMES):
        sota_str = f"{sota_vals[i]:.3f}" if sota_vals else "N/A"
        lines.append(f"{name:<15} | {means[i]:.4f} ± {stds[i]:.4f} | {sota_str:<10}")
        
    lines.append("-" * 85)
    lines.append(f"Overall AvgImp: {overall_avg_imp:.2%}")
    lines.append("-" * 85)
    lines.append("Runs Results:")
    for run_idx, metrics in enumerate(all_run_metrics):
        lines.append(f"  Run {run_idx}: {metrics}")
    
    final_content = "\n".join(lines)
    print("\n" + final_content)

    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(final_content)
    print(f"\n💾 Saved result.txt to {txt_path}")
    
    # 保存最优模型checkpoint路径
    if best_model_paths:
        with open(best_model_path, 'w', encoding='utf-8') as f:
            for path in best_model_paths:
                f.write(f"{path}\n")
        print(f"💾 Saved best_model_path.txt to {best_model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Dataset name")
    parser.add_argument("--device", required=True, help="Device ID, e.g., 'cuda:0'")
    args = parser.parse_args()
    main(args.dataset, args.device)
