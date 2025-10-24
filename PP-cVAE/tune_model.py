import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
import itertools
from typing import Dict, List, Any, Optional
from config import Config, ExperimentConfig
from experiment import run_experiment
from utils import set_seed, create_experiment_dir
class HyperparameterTuner:
    def __init__(self, base_config=None, save_dir='hyperparameter_tuning'):
        self.base_config = base_config or Config()
        self.save_dir = save_dir
        self.results = []
        os.makedirs(save_dir, exist_ok=True)
    def grid_search(self, param_grid: Dict[str, List], max_trials: int = None, 
                   metric: str = 'test_r2', higher_better: bool = True):
        print(f"\n=== Grid Search Hyperparameter Tuning ===")
        print(f"Parameter grid: {param_grid}")
        print(f"Optimization metric: {metric}")
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(itertools.product(*param_values))
        if max_trials and len(combinations) > max_trials:
            print(f"Limiting trials from {len(combinations)} to {max_trials}")
            import random
            random.shuffle(combinations)
            combinations = combinations[:max_trials]
        print(f"Running {len(combinations)} trials...")
        best_score = float('-inf') if higher_better else float('inf')
        best_params = None
        best_result = None
        for i, combination in enumerate(combinations):
            print(f"\n--- Trial {i+1}/{len(combinations)} ---")
            params = dict(zip(param_names, combination))
            print(f"Parameters: {params}")
            experiment_name = f'grid_search_trial_{i+1:03d}'
            result = run_experiment(
                experiment_name=experiment_name,
                config_overrides=params,
                save_results=True,
                base_config=self.base_config
            )
            if result:
                trial_result = {
                    'trial': i + 1,
                    'params': params,
                    'result': result,
                    'metric_value': result.get(metric, 0)
                }
                self.results.append(trial_result)
                current_score = result.get(metric, 0)
                if (higher_better and current_score > best_score) or \
                   (not higher_better and current_score < best_score):
                    best_score = current_score
                    best_params = params
                    best_result = result
                print(f"Result: {metric} = {current_score:.4f}")
                print(f"Best so far: {metric} = {best_score:.4f}")
            else:
                print(f"Trial failed!")
        self._save_results('grid_search')
        print(f"\n=== Grid Search Completed ===")
        print(f"Best parameters: {best_params}")
        print(f"Best {metric}: {best_score:.4f}")
        return best_params, best_result, self.results
    def random_search(self, param_distributions: Dict[str, Any], n_trials: int = 20,
                     metric: str = 'test_r2', higher_better: bool = True):
        print(f"\n=== Random Search Hyperparameter Tuning ===")
        print(f"Parameter distributions: {param_distributions}")
        print(f"Number of trials: {n_trials}")
        print(f"Optimization metric: {metric}")
        import random
        best_score = float('-inf') if higher_better else float('inf')
        best_params = None
        best_result = None
        for i in range(n_trials):
            print(f"\n--- Trial {i+1}/{n_trials} ---")
            params = {}
            for param_name, distribution in param_distributions.items():
                if isinstance(distribution, list):
                    params[param_name] = random.choice(distribution)
                elif isinstance(distribution, tuple) and len(distribution) == 2:
                    min_val, max_val = distribution
                    if isinstance(min_val, int) and isinstance(max_val, int):
                        params[param_name] = random.randint(min_val, max_val)
                    else:
                        params[param_name] = random.uniform(min_val, max_val)
                elif callable(distribution):
                    params[param_name] = distribution()
            print(f"Parameters: {params}")
            experiment_name = f'random_search_trial_{i+1:03d}'
            result = run_experiment(
                experiment_name=experiment_name,
                config_overrides=params,
                save_results=True,
                base_config=self.base_config
            )
            if result:
                trial_result = {
                    'trial': i + 1,
                    'params': params,
                    'result': result,
                    'metric_value': result.get(metric, 0)
                }
                self.results.append(trial_result)
                current_score = result.get(metric, 0)
                if (higher_better and current_score > best_score) or \
                   (not higher_better and current_score < best_score):
                    best_score = current_score
                    best_params = params
                    best_result = result
                print(f"Result: {metric} = {current_score:.4f}")
                print(f"Best so far: {metric} = {best_score:.4f}")
            else:
                print(f"Trial failed!")
        self._save_results('random_search')
        print(f"\n=== Random Search Completed ===")
        print(f"Best parameters: {best_params}")
        print(f"Best {metric}: {best_score:.4f}")
        return best_params, best_result, self.results
    def bayesian_optimization(self, param_space: Dict[str, Any], n_trials: int = 20,
                            metric: str = 'test_r2', higher_better: bool = True):
        try:
            from skopt import gp_minimize
            from skopt.space import Real, Integer, Categorical
            from skopt.utils import use_named_args
        except ImportError:
            print("scikit-optimize not installed. Please install it for Bayesian optimization.")
            print("pip install scikit-optimize")
            return None, None, []
        print(f"\n=== Bayesian Optimization Hyperparameter Tuning ===")
        print(f"Parameter space: {param_space}")
        print(f"Number of trials: {n_trials}")
        print(f"Optimization metric: {metric}")
        dimensions = []
        param_names = []
        for param_name, param_def in param_space.items():
            param_names.append(param_name)
            if param_def['type'] == 'real':
                dimensions.append(Real(param_def['low'], param_def['high'], name=param_name))
            elif param_def['type'] == 'int':
                dimensions.append(Integer(param_def['low'], param_def['high'], name=param_name))
            elif param_def['type'] == 'categorical':
                dimensions.append(Categorical(param_def['choices'], name=param_name))
        @use_named_args(dimensions)
        def objective(**params):
            print(f"\nTrial {len(self.results) + 1}: {params}")
            experiment_name = f'bayesian_trial_{len(self.results) + 1:03d}'
            result = run_experiment(
                experiment_name=experiment_name,
                config_overrides=params,
                save_results=True,
                base_config=self.base_config
            )
            if result:
                trial_result = {
                    'trial': len(self.results) + 1,
                    'params': params,
                    'result': result,
                    'metric_value': result.get(metric, 0)
                }
                self.results.append(trial_result)
                score = result.get(metric, 0)
                print(f"Result: {metric} = {score:.4f}")
                return -score if higher_better else score
            else:
                print("Trial failed!")
                return float('inf')
        result = gp_minimize(
            func=objective,
            dimensions=dimensions,
            n_calls=n_trials,
            random_state=42
        )
        best_params = dict(zip(param_names, result.x))
        best_score = -result.fun if higher_better else result.fun
        best_result = None
        for trial in self.results:
            if trial['metric_value'] == best_score:
                best_result = trial['result']
                break
        self._save_results('bayesian_optimization')
        print(f"\n=== Bayesian Optimization Completed ===")
        print(f"Best parameters: {best_params}")
        print(f"Best {metric}: {best_score:.4f}")
        return best_params, best_result, self.results
    def _save_results(self, method_name: str):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_path = os.path.join(self.save_dir, f'{method_name}_results_{timestamp}.json')
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        if self.results:
            df_data = []
            for trial in self.results:
                row = {'trial': trial['trial']}
                row.update(trial['params'])
                row.update({
                    'test_r2': trial['result'].get('test_r2', 0),
                    'val_r2': trial['result'].get('val_r2', 0),
                    'test_rmse': trial['result'].get('test_rmse', 0),
                    'val_rmse': trial['result'].get('val_rmse', 0)
                })
                df_data.append(row)
            df = pd.DataFrame(df_data)
            csv_path = os.path.join(self.save_dir, f'{method_name}_summary_{timestamp}.csv')
            df.to_csv(csv_path, index=False)
            print(f"\nResults saved to:")
            print(f"  Detailed: {results_path}")
            print(f"  Summary: {csv_path}")
def main():
    import argparse
    parser = argparse.ArgumentParser(description='Hyperparameter tuning for VAE-CNN')
    parser.add_argument('--method', choices=['grid', 'random', 'bayesian'], 
                       default='grid', help='Tuning method')
    parser.add_argument('--trials', type=int, default=20, 
                       help='Number of trials (for random/bayesian)')
    parser.add_argument('--metric', type=str, default='test_r2',
                       help='Optimization metric')
    parser.add_argument('--save_dir', type=str, default='hyperparameter_tuning',
                       help='Save directory')
    parser.add_argument('--fast_mode', action='store_true',
                       help='Enable fast tuning mode (fewer epochs for quick testing)')
    parser.add_argument('--fast_epochs', type=int, default=10,
                       help='Number of epochs for fast mode')
    args = parser.parse_args()
    set_seed(42)
    base_config = Config()
    if args.fast_mode:
        base_config.FAST_TUNING_MODE = True
        base_config.FAST_TUNING_EPOCHS = args.fast_epochs
        base_config.USE_MIXED_PRECISION = True  
        print(f"Fast tuning mode enabled: {args.fast_epochs} epochs per trial")
        print("Mixed precision training enabled for acceleration")
    tuner = HyperparameterTuner(base_config=base_config, save_dir=args.save_dir)
    if args.method == 'grid':
        param_grid = {
            'learning_rate': [1e-4, 5e-4, 1e-3, 5e-3],
            'latent_dim': [64, 128, 256],
            'batch_size': [8, 16, 32],
            'vae_beta': [0.01, 0.1, 1.0],
            'dropout_rate': [0.1, 0.3, 0.5]
        }
        best_params, best_result, results = tuner.grid_search(
            param_grid, max_trials=args.trials, metric=args.metric
        )
    elif args.method == 'random':
        param_distributions = {
            'learning_rate': [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
            'latent_dim': [32, 64, 128, 256, 512],
            'batch_size': [4, 8, 16, 32, 64],
            'vae_beta': (0.001, 10.0),  
            'dropout_rate': (0.0, 0.7),  
            'vae_weight': (0.001, 1.0)  
        }
        best_params, best_result, results = tuner.random_search(
            param_distributions, n_trials=args.trials, metric=args.metric
        )
    elif args.method == 'bayesian':
        param_space = {
            'learning_rate': {'type': 'real', 'low': 1e-5, 'high': 1e-2},
            'latent_dim': {'type': 'int', 'low': 32, 'high': 512},
            'batch_size': {'type': 'categorical', 'choices': [4, 8, 16, 32, 64]},
            'vae_beta': {'type': 'real', 'low': 0.001, 'high': 10.0},
            'dropout_rate': {'type': 'real', 'low': 0.0, 'high': 0.7}
        }
        best_params, best_result, results = tuner.bayesian_optimization(
            param_space, n_trials=args.trials, metric=args.metric
        )
    if best_params:
        print(f"\n=== Final Best Configuration ===")
        print(f"Parameters: {best_params}")
        print(f"Test R² Score: {best_result['test_r2']:.4f}")
        print(f"Test RMSE: {best_result['test_rmse']:.4f}")
        best_config_path = os.path.join(args.save_dir, 'best_config.json')
        with open(best_config_path, 'w') as f:
            json.dump({
                'method': args.method,
                'best_params': best_params,
                'best_result': best_result,
                'metric': args.metric
            }, f, indent=2, default=str)
        print(f"Best configuration saved to: {best_config_path}")
    return 0
if __name__ == '__main__':
    exit(main())