"""批量实验脚本 - 运行多个实验并比较结果"""

import os
import json
import pandas as pd
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

from config import EXPERIMENT_CONFIGS
from experiment import run_experiment
from visualization import plot_experiment_comparison
from utils import set_seed

def run_single_experiment_wrapper(args):
    """包装函数用于多进程"""
    experiment_name, config_overrides, save_results = args
    return run_experiment(experiment_name, config_overrides, save_results)

def run_multiple_experiments(experiment_list=None, parallel=False, save_results=True):
    """运行多个实验"""
    
    if experiment_list is None:
        experiment_list = list(EXPERIMENT_CONFIGS.keys())
    
    print(f"\n=== Running {len(experiment_list)} Experiments ===")
    print(f"Experiments: {experiment_list}")
    print(f"Parallel execution: {parallel}")
    
    results = {}
    
    if parallel and len(experiment_list) > 1:
        # 并行执行
        print("\nRunning experiments in parallel...")
        
        # 准备参数
        args_list = [(name, None, save_results) for name in experiment_list]
        
        # 使用进程池
        with ProcessPoolExecutor(max_workers=min(len(experiment_list), mp.cpu_count())) as executor:
            experiment_results = list(executor.map(run_single_experiment_wrapper, args_list))
        
        # 整理结果
        for i, result in enumerate(experiment_results):
            if result:
                results[experiment_list[i]] = result
    else:
        # 串行执行
        print("\nRunning experiments sequentially...")
        
        for experiment_name in experiment_list:
            print(f"\n{'='*50}")
            result = run_experiment(experiment_name, save_results=save_results)
            if result:
                results[experiment_name] = result
    
    return results

def create_comparison_report(results, save_dir='experiment_comparison'):
    """创建实验比较报告"""
    
    if not results:
        print("No results to compare")
        return None
    
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 创建结果表格
    comparison_data = []
    for exp_name, result in results.items():
        comparison_data.append({
            'Experiment': exp_name,
            'Train_R2': result['train_r2'],
            'Val_R2': result['val_r2'],
            'Test_R2': result['test_r2'],
            'Train_RMSE': result['train_rmse'],
            'Val_RMSE': result['val_rmse'],
            'Test_RMSE': result['test_rmse'],
            'Model_Parameters': result['model_parameters'],
            'Best_Val_Loss': result['best_val_loss']
        })
    
    # 保存为CSV
    df = pd.DataFrame(comparison_data)
    csv_path = os.path.join(save_dir, f'experiment_comparison_{timestamp}.csv')
    df.to_csv(csv_path, index=False)
    
    # 保存为JSON
    json_path = os.path.join(save_dir, f'experiment_results_{timestamp}.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # 创建比较图表
    plot_data = {exp: {'r2': res['test_r2'], 'mse': res['test_rmse']**2, 'rmse': res['test_rmse']} 
                for exp, res in results.items()}
    
    plot_path = os.path.join(save_dir, f'experiment_comparison_{timestamp}.png')
    plot_experiment_comparison(plot_data, save_path=plot_path)
    
    # 打印总结
    print(f"\n=== Experiment Comparison Report ===")
    print(f"Results saved to: {save_dir}")
    print(f"\nSummary Table:")
    print(df.round(4).to_string(index=False))
    
    # 找出最佳实验
    best_test_r2 = df.loc[df['Test_R2'].idxmax()]
    best_val_r2 = df.loc[df['Val_R2'].idxmax()]
    
    print(f"\nBest Test R² Score: {best_test_r2['Experiment']} ({best_test_r2['Test_R2']:.4f})")
    print(f"Best Val R² Score: {best_val_r2['Experiment']} ({best_val_r2['Val_R2']:.4f})")
    
    return {
        'csv_path': csv_path,
        'json_path': json_path,
        'plot_path': plot_path,
        'summary_df': df,
        'best_test': best_test_r2.to_dict(),
        'best_val': best_val_r2.to_dict()
    }

def run_hyperparameter_search(param_grid, base_experiment='baseline', max_experiments=10):
    """运行超参数搜索"""
    
    print(f"\n=== Hyperparameter Search ===")
    print(f"Base experiment: {base_experiment}")
    print(f"Parameter grid: {param_grid}")
    
    # 生成参数组合
    import itertools
    
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    
    # 生成所有组合
    combinations = list(itertools.product(*param_values))
    
    if len(combinations) > max_experiments:
        print(f"Too many combinations ({len(combinations)}), limiting to {max_experiments}")
        # 随机选择
        import random
        random.shuffle(combinations)
        combinations = combinations[:max_experiments]
    
    results = {}
    
    for i, combination in enumerate(combinations):
        # 创建配置覆盖
        config_overrides = dict(zip(param_names, combination))
        experiment_name = f"hypersearch_{i+1:02d}"
        
        print(f"\nRunning {experiment_name}: {config_overrides}")
        
        result = run_experiment(
            experiment_name=experiment_name,
            config_overrides=config_overrides,
            save_results=True
        )
        
        if result:
            result['param_combination'] = config_overrides
            results[experiment_name] = result
    
    # 创建超参数搜索报告
    if results:
        report = create_comparison_report(results, save_dir='hyperparameter_search')
        
        # 分析最佳参数
        best_result = max(results.values(), key=lambda x: x['test_r2'])
        print(f"\nBest hyperparameters: {best_result['param_combination']}")
        print(f"Best test R² score: {best_result['test_r2']:.4f}")
        
        return results, report
    
    return None, None

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run multiple VAE-CNN experiments')
    parser.add_argument('--experiments', '-e', nargs='+', 
                       help='List of experiments to run')
    parser.add_argument('--all', action='store_true', 
                       help='Run all predefined experiments')
    parser.add_argument('--parallel', action='store_true', 
                       help='Run experiments in parallel')
    parser.add_argument('--hypersearch', action='store_true',
                       help='Run hyperparameter search')
    parser.add_argument('--no_save', action='store_true', 
                       help='Do not save results')
    
    args = parser.parse_args()
    
    set_seed(42)
    
    if args.hypersearch:
        # 超参数搜索
        param_grid = {
            'learning_rate': [1e-4, 5e-4, 1e-3, 5e-3],
            'latent_dim': [64, 128, 256],
            'batch_size': [8, 16, 32]
        }
        
        results, report = run_hyperparameter_search(param_grid, max_experiments=12)
        
    elif args.all:
        # 运行所有预定义实验
        results = run_multiple_experiments(
            experiment_list=None,
            parallel=args.parallel,
            save_results=not args.no_save
        )
        
        if results:
            report = create_comparison_report(results)
    
    elif args.experiments:
        # 运行指定实验
        results = run_multiple_experiments(
            experiment_list=args.experiments,
            parallel=args.parallel,
            save_results=not args.no_save
        )
        
        if results:
            report = create_comparison_report(results)
    
    else:
        # 默认运行baseline实验
        print("No experiments specified. Running baseline experiment.")
        result = run_experiment('baseline', save_results=not args.no_save)
        if result:
            print(f"\nBaseline experiment completed.")
            print(f"Test R² Score: {result['test_r2']:.4f}")
    
    return 0

if __name__ == '__main__':
    exit(main())