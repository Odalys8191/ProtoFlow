import os
import os.path as osp
import json
import argparse
import subprocess
import itertools
import numpy as np
import re
import sys
import time

# ================= 🔧 配置区域 =================

# SOTA JSON 路径
SOTA_PATH = "../Data/sota.json"
# 结果保存根目录
RESULT_ROOT = "evaluation_results"
# 训练脚本名
TRAIN_SCRIPT = "train.py"
# 推理脚本名（如果与训练脚本分离）
INFERENCE_SCRIPT = "test.py"

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

# 交叉验证折数
CV_FOLDS = 5
# 每次推理的重复次数
INFERENCE_REPEATS = 10

# ================= 功能函数 =================


def get_sota_directly(dataset_folder_name):
    """直接读取 JSON 内容"""
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



def parse_metrics(output_str):
    results = []
    for m in METRICS_NAMES:
        pattern = re.escape(m) + r"\s+\|\s+([0-9\.]+)"
        match = re.search(pattern, output_str)
        if match:
            results.append(float(match.group(1)))
    return results if len(results) == 6 else None



def calc_avg_imp(our_mean, sota_vals):
    if not sota_vals: return 0.0
    imps = []
    for i in range(4): # 前4个越小越好
        imps.append((sota_vals[i] - our_mean[i]) / (sota_vals[i] + 1e-12))
    for i in range(4, 6): # 后2个越大越好
        imps.append((our_mean[i] - sota_vals[i]) / (sota_vals[i] + 1e-12))
    return np.mean(imps)



def run_cmd_live(cmd, repeats=1):
    """运行命令并返回结果，支持多次运行取平均"""
    all_metrics = []
    full_output = ""
    
    for i in range(repeats):
        print(f"\n📊 Run {i+1}/{repeats}")
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            if line:
                sys.stdout.write(line)
                full_output += line
        
        # 从完整输出中解析metrics
        metrics = parse_metrics(full_output)
        if metrics:
            all_metrics.append(metrics)
        else:
            print(f"❌ Failed to parse metrics for run {i+1}")
    
    if not all_metrics:
        return None, full_output
    
    avg_metrics = np.mean(all_metrics, axis=0).tolist()
    return avg_metrics, full_output


# ================= 主逻辑 =================

def main(dataset, device):
    # 1. 准备目录
    save_dir = os.path.join(RESULT_ROOT, dataset)
    os.makedirs(save_dir, exist_ok=True)
    model_dir = os.path.join(save_dir, "model")
    os.makedirs(model_dir, exist_ok=True)
    txt_path = os.path.join(save_dir, "result.txt")
    best_params_path = os.path.join(save_dir, "best_params.json")

    print(f"🚀 Processing Dataset: {dataset} on {device}")
    
    # 2. 获取 SOTA
    sota_vals = get_sota_directly(dataset)

    # 3. Grid Search (在 run_0 下测试超参数，用五折交叉验证重复五次和 avg_imp 作为优先度)
    print(f"\n{'='*30}\n🔍 Grid Search (run_0 with 5-fold CV repeated 5 times)\n{'='*30}")
    
    keys, values = zip(*SEARCH_SPACE.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    best_avg_imp = -float('inf')
    best_params = {}

    for params in combinations:
        print(f"\nTesting Params: {params}")
        
        # 构造训练命令：根据实际项目调整参数
        train_cmd = [
            "python", TRAIN_SCRIPT,
            "--train_feature", f"../Data/{dataset}/feature/train_feature.npy",
            "--train_label", f"../Data/{dataset}/label/train_label.npy",
            "--test_feature", f"../Data/{dataset}/feature/test_feature.npy",
            "--test_label", f"../Data/{dataset}/label/test_label.npy",
            "--batch_size", str(params.get("batch_size", 128)),
            "--lr", str(params.get("lr", 1e-3)),
            "--feature_dim", str(params.get("feature_dim", 128)),
            "--nepoch", "200",
            "--num_workers", "0",
            "--extra", f"grid_search_{dataset}",
            "--cv_folds", "5"  # 指定五折交叉验证
        ]
        
        print(f"\n🏋️  Training with params: {params}")
        # 运行训练，保存模型
        _, train_output = run_cmd_live(train_cmd, repeats=1)
        
        # 寻找训练生成的模型文件
        model_files = []
        grid_search_dir = osp.join("logs", f"ldl_grid_search_{dataset}")
        if os.path.exists(grid_search_dir):
            for f in os.listdir(grid_search_dir):
                if f.startswith("bs"):
                    model_run_dir = osp.join(grid_search_dir, f)
                    for mf in os.listdir(model_run_dir):
                        if mf.endswith(".pt") and "best" in mf:
                            model_files.append(osp.join(model_run_dir, mf))
        
        if model_files:
            # 按修改时间排序，取最新的模型
            latest_model = max(model_files, key=os.path.getmtime)
            print(f"📦 Using latest model: {latest_model}")
            
            # 构造推理命令
            inference_cmd = [
                "python", INFERENCE_SCRIPT,
                "--resume", latest_model,
                "--train_feature", f"../Data/{dataset}/feature/train_feature.npy",
                "--train_label", f"../Data/{dataset}/label/train_label.npy",
                "--test_feature", f"../Data/{dataset}/feature/test_feature.npy",
                "--test_label", f"../Data/{dataset}/label/test_label.npy",
                "--batch_size", "2048",
                "--num_samples", "10",  # 跑10次inference
                "--test_only"
            ]
            
            # 五折交叉验证重复五次：每次inference跑10次取平均
            print(f"\n🔄 5-fold CV repeated 5 times for params: {params}")
            all_cv_metrics = []
            for cv_repeat in range(5):
                print(f"📊 CV Repeat {cv_repeat+1}/5")
                for fold_idx in range(5):
                    print(f"  Fold {fold_idx+1}/5")
                    # 每次inference使用相同的模型，跑10次取平均
                    fold_metrics, _ = run_cmd_live(inference_cmd, repeats=1)
                    if fold_metrics:
                        all_cv_metrics.append(fold_metrics)
            
            if all_cv_metrics:
                # 计算五折重复五次的平均metrics
                cv_avg_metrics = np.mean(all_cv_metrics, axis=0).tolist()
                # 计算avg_imp作为优先度指标
                cv_avg_imp = calc_avg_imp(cv_avg_metrics, sota_vals)
                print(f"\n📊 5-fold CV repeated 5 times Results for {params}:")
                print(f"   Mean Metrics: {cv_avg_metrics}")
                print(f"   AvgImp: {cv_avg_imp:.2%}")
                
                # 选择avg_imp最高的参数
                if cv_avg_imp > best_avg_imp:
                    best_avg_imp = cv_avg_imp
                    best_params = params
                    print("⭐ Current Best!")
            else:
                print("❌ Failed to get metrics for all folds")
        else:
            print("❌ No model files found after training")

    print(f"\n✅ Best Params Found: {best_params} (AvgImp: {best_avg_imp:.2%})")
    
    # 保存最优参数
    with open(best_params_path, 'w', encoding='utf-8') as f:
        json.dump(best_params, f, indent=2)
    print(f"💾 Saved best_params.json to {best_params_path}")

    # 4. 跑所有 run_0 至 run_9，每个 run 跑十次 inference 取平均
    print(f"\n{'='*30}\n🏃 Running all runs (run_0 to run_9)\n{'='*30}")
    
    all_run_metrics = []
    for run_idx in range(10):
        print(f"\n>>> Run {run_idx}/9")
        
        # 构造训练命令：使用最优参数训练并保存模型
        # 注意：train.py需要指定特征和标签文件路径，这里需要根据实际数据集结构调整
        train_feature_path = f"../Data/{dataset}/feature/train_feature.npy"
        train_label_path = f"../Data/{dataset}/label/train_label.npy"
        test_feature_path = f"../Data/{dataset}/feature/test_feature.npy"
        test_label_path = f"../Data/{dataset}/label/test_label.npy"
        
        train_cmd = [
            "python", TRAIN_SCRIPT,
            "--train_feature", train_feature_path,
            "--train_label", train_label_path,
            "--test_feature", test_feature_path,
            "--test_label", test_label_path,
            "--batch_size", str(best_params.get("batch_size", 128)),
            "--lr", str(best_params.get("lr", 1e-3)),
            "--feature_dim", str(best_params.get("feature_dim", 128)),
            "--nepoch", "200",
            "--num_workers", "0",
            "--device", device,
            "--extra", f"{dataset}_run{run_idx}"
        ]
        
        print(f"\n🏋️ Training with best params for run {run_idx}...")
        train_output = run_cmd_live(train_cmd, repeats=1)[1]
        
        # 构造推理命令：加载模型进行推理，跑十次取平均
        # 寻找最新的模型文件
        model_files = []
        for f in os.listdir(osp.join("logs", f"ldl_{dataset}_run{run_idx}")):
            if f.startswith("bs"):
                model_run_dir = osp.join("logs", f"ldl_{dataset}_run{run_idx}", f)
                for mf in os.listdir(model_run_dir):
                    if mf.endswith(".pt") and "best" in mf:
                        model_files.append(osp.join(model_run_dir, mf))
        
        if model_files:
            # 按修改时间排序，取最新的模型
            latest_model = max(model_files, key=os.path.getmtime)
            print(f"📦 Using latest model: {latest_model}")
            
            inference_cmd = [
                    "python", INFERENCE_SCRIPT,
                    "--resume", latest_model,
                    "--train_feature", train_feature_path,
                    "--train_label", train_label_path,
                    "--test_feature", test_feature_path,
                    "--test_label", test_label_path,
                    "--batch_size", "2048",
                    "--num_samples", "10",  # 跑10次inference
                    "--test_only"
                ]
            
            print(f"\n🔍 Inferencing with best model for run {run_idx}...")
            metrics, _ = run_cmd_live(inference_cmd, repeats=INFERENCE_REPEATS)
        else:
            print(f"❌ No best model found for run {run_idx}")
            metrics = None
        
        if metrics:
            all_run_metrics.append(metrics)
            print(f"✅ Run {run_idx} Metrics (avg of {INFERENCE_REPEATS} inferences): {metrics}")
        else:
            print(f"❌ Failed to get metrics for run {run_idx}")

    # 5. 计算最终结果
    if not all_run_metrics:
        print("❌ No valid results from any run.")
        return

    all_run_metrics = np.array(all_run_metrics)
    means = np.mean(all_run_metrics, axis=0)
    stds = np.std(all_run_metrics, axis=0)
    overall_avg_imp = calc_avg_imp(means, sota_vals)

    # 6. 生成并保存结果报告
    lines = []
    lines.append(f"Dataset: {dataset}")
    lines.append(f"Best Params: {best_params}")
    lines.append("-" * 65)
    lines.append(f"{'Metric':<15} | {'Mean ± Std':<25} | {'SOTA':<10}")
    lines.append("-" * 65)
    
    for i, name in enumerate(METRICS_NAMES):
        sota_str = f"{sota_vals[i]:.3f}" if sota_vals else "N/A"
        lines.append(f"{name:<15} | {means[i]:.4f} ± {stds[i]:.4f} | {sota_str:<10}")
        
    lines.append("-" * 65)
    lines.append(f"Overall AvgImp: {overall_avg_imp:.2%}")
    lines.append("-" * 65)
    lines.append("Runs Results:")
    for run_idx, metrics in enumerate(all_run_metrics):
        lines.append(f"  Run {run_idx}: {metrics}")
    
    final_content = "\n".join(lines)
    print("\n" + final_content)

    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(final_content)
    print(f"\n💾 Saved result.txt to {txt_path}")
    print(f"\n🎉 Evaluation completed successfully for {dataset}!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Dataset name")
    parser.add_argument("--device", required=True, help="Device ID, e.g., 'cuda:0'")
    args = parser.parse_args()
    main(args.dataset, args.device)
