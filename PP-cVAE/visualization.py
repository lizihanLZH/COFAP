import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import r2_score
import os
plt.style.use('default')
sns.set_palette("husl")
def plot_training_history(train_losses, val_losses, val_r2_scores, save_path=None):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].plot(train_losses, label='Train Loss', linewidth=2)
    axes[0].plot(val_losses, label='Val Loss', linewidth=2)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(val_r2_scores, color='green', linewidth=2)
    axes[1].set_title('Validation R² Score', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('R² Score')
    axes[1].grid(True, alpha=0.3)
    if len(train_losses) > 50:
        start_idx = len(train_losses) - 50
        axes[2].plot(range(start_idx, len(train_losses)), train_losses[start_idx:], 
                    label='Train Loss', linewidth=2)
        axes[2].plot(range(start_idx, len(val_losses)), val_losses[start_idx:], 
                    label='Val Loss', linewidth=2)
        axes[2].set_title('Learning Curve (Last 50 Epochs)', fontsize=14, fontweight='bold')
    else:
        axes[2].plot(train_losses, label='Train Loss', linewidth=2)
        axes[2].plot(val_losses, label='Val Loss', linewidth=2)
        axes[2].set_title('Learning Curve', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Loss')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'Training history plot saved to {save_path}')
    return fig
def plot_predictions(predictions, targets, save_path=None, title='Predictions vs True Values'):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    axes[0].scatter(targets, predictions, alpha=0.6, s=20)
    min_val = min(min(targets), min(predictions))
    max_val = max(max(targets), max(predictions))
    axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    r2 = r2_score(targets, predictions)
    axes[0].set_title(f'{title}\nR² = {r2:.4f}', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('True Values')
    axes[0].set_ylabel('Predictions')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    residuals = np.array(predictions) - np.array(targets)
    axes[1].scatter(targets, residuals, alpha=0.6, s=20)
    axes[1].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[1].set_title('Residuals vs True Values', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('True Values')
    axes[1].set_ylabel('Residuals (Predicted - True)')
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'Predictions plot saved to {save_path}')
    return fig
def plot_error_distribution(predictions, targets, save_path=None):
    residuals = np.array(predictions) - np.array(targets)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].hist(residuals, bins=50, alpha=0.7, edgecolor='black')
    axes[0].axvline(x=0, color='r', linestyle='--', linewidth=2)
    axes[0].set_title('Residuals Distribution', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Residuals')
    axes[0].set_ylabel('Frequency')
    axes[0].grid(True, alpha=0.3)
    abs_errors = np.abs(residuals)
    axes[1].hist(abs_errors, bins=50, alpha=0.7, color='orange', edgecolor='black')
    axes[1].set_title('Absolute Error Distribution', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Absolute Error')
    axes[1].set_ylabel('Frequency')
    axes[1].grid(True, alpha=0.3)
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[2])
    axes[2].set_title('Q-Q Plot (Normal Distribution)', fontsize=14, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'Error distribution plot saved to {save_path}')
    return fig
def plot_experiment_comparison(experiment_results, save_path=None):
    if not experiment_results:
        print("No experiment results to plot")
        return None
    experiments = list(experiment_results.keys())
    r2_scores = [experiment_results[exp]['r2'] for exp in experiments]
    mse_scores = [experiment_results[exp]['mse'] for exp in experiments]
    rmse_scores = [experiment_results[exp]['rmse'] for exp in experiments]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    bars1 = axes[0].bar(experiments, r2_scores, alpha=0.7)
    axes[0].set_title('R² Score Comparison', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('R² Score')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(True, alpha=0.3)
    for bar, score in zip(bars1, r2_scores):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom')
    bars2 = axes[1].bar(experiments, mse_scores, alpha=0.7, color='orange')
    axes[1].set_title('MSE Comparison', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('MSE')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(True, alpha=0.3)
    bars3 = axes[2].bar(experiments, rmse_scores, alpha=0.7, color='green')
    axes[2].set_title('RMSE Comparison', fontsize=14, fontweight='bold')
    axes[2].set_ylabel('RMSE')
    axes[2].tick_params(axis='x', rotation=45)
    axes[2].grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'Experiment comparison plot saved to {save_path}')
    return fig
def create_comprehensive_report(train_losses, val_losses, val_r2_scores, 
                              test_predictions, test_targets, 
                              experiment_name='experiment', save_dir='results'):
    os.makedirs(save_dir, exist_ok=True)
    history_fig = plot_training_history(train_losses, val_losses, val_r2_scores)
    history_path = os.path.join(save_dir, f'{experiment_name}_training_history.png')
    history_fig.savefig(history_path, dpi=300, bbox_inches='tight')
    plt.close(history_fig)
    pred_fig = plot_predictions(test_predictions, test_targets)
    pred_path = os.path.join(save_dir, f'{experiment_name}_predictions.png')
    pred_fig.savefig(pred_path, dpi=300, bbox_inches='tight')
    plt.close(pred_fig)
    error_fig = plot_error_distribution(test_predictions, test_targets)
    error_path = os.path.join(save_dir, f'{experiment_name}_error_distribution.png')
    error_fig.savefig(error_path, dpi=300, bbox_inches='tight')
    plt.close(error_fig)
    print(f'Comprehensive report saved to {save_dir}')
    return history_path, pred_path, error_path
def show_plots():
    plt.show()