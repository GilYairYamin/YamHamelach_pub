from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import os
import yaml
import numpy as np

class ExperimentConfig:
    def __init__(self, config):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_name = config['train_params'].get('task_name', 'q1')
        
        # Create the main results directory
        self.base_results_dir = "results"
        os.makedirs(self.base_results_dir, exist_ok=True)
        
        # Create the specific experiment directory and subdirectories
        self.results_dir = os.path.join(self.base_results_dir, f"{self.experiment_name}_{self.timestamp}")
        self.metrics_dir = os.path.join(self.results_dir, "metrics")
        self.checkpoints_dir = os.path.join(self.results_dir, "checkpoints")
        self.logs_dir = os.path.join(self.results_dir, "logs")
        for directory in [self.results_dir, self.metrics_dir, self.checkpoints_dir, self.logs_dir]:
            os.makedirs(directory, exist_ok=True)
        
        # Save the configuration parameters
        self.dataset_params = config['dataset_params']
        self.model_params = config['model_params']
        self.train_params = config['train_params']
        self.save_config()
        
        # Create a log file to track the execution
        self.log_file = os.path.join(self.logs_dir, "training.log")
        with open(self.log_file, "w") as f:
            f.write(f"Starting experiment: {self.experiment_name}\n")
            f.write(f"Timestamp: {self.timestamp}\n")
            f.write("-" * 50 + "\n")

    def save_config(self):
        config_path = os.path.join(self.results_dir, "config.yaml")
        with open(config_path, "w") as f:
            yaml.dump(self.__dict__, f)

    def log_message(self, message):
        """Add a message to the logger"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_file, "a") as f:
            f.write(f"[{timestamp}] {message}\n")


class MetricsTracker:
    def __init__(self, config):
        self.config = config
        self.metrics = {}
        self.history = {
            'rpn_class_loss': [],
            'rpn_bbox_loss': [],
            'det_class_loss': [],
            'det_bbox_loss': [],
            'total_loss': [],
        }
        self.results_dir = config.results_dir
        self.metrics_dir = config.metrics_dir

    def update(self, metrics):
        for key, value in metrics.items():
            self.history[key].append(float(value))
    
    def plot_metrics(self, epoch):
        # Style configuration
        plt.rcParams['figure.figsize'] = [15, 20]
        plt.rcParams['axes.grid'] = True
        plt.rcParams['axes.spines.top'] = False
        plt.rcParams['axes.spines.right'] = False
        
        # Create figure with hyperparameters
        fig, axes = plt.subplots(3, 1)
        fig.suptitle(f"Training Metrics - {self.config.experiment_name}\n" + f"Epoch {epoch}", fontsize=16)
        
        # Define a color palette
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        # Plot different losses with logarithmic scale
        axes[0].semilogy(self.history['total_loss'], label='Total Loss', color=colors[0], linewidth=2)
        axes[0].set_title('Total Loss (log scale)')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        
        # RPN losses
        axes[1].semilogy(self.history['rpn_class_loss'], label='RPN Class', color=colors[1], linewidth=2)
        axes[1].semilogy(self.history['rpn_bbox_loss'], label='RPN Box', color=colors[2], linewidth=2)
        axes[1].set_title('RPN Losses (log scale)')
        axes[1].legend()
        
        # Detection losses
        axes[2].semilogy(self.history['det_class_loss'], label='Detection Class', color=colors[3], linewidth=2)
        axes[2].semilogy(self.history['det_bbox_loss'], label='Detection Box', color=colors[4], linewidth=2)
        axes[2].set_title('Detection Losses (log scale)')
        axes[2].legend()
        
        # Add a grid to all subplots
        for ax in axes.flat:
            ax.grid(True, linestyle='--', alpha=0.7)
        
        # Adjust spacing
        plt.tight_layout()
        
        # Save
        plt.savefig(os.path.join(self.metrics_dir, f'metrics_epoch_{epoch}.png'), bbox_inches='tight', dpi=300)
        plt.close()
        
        # Log the creation of the plot
        self.config.log_message(f"Created metrics plot for epoch {epoch}")


def plot_precision_recall_curve(ap, precisions, recalls, method='area', label_name='', save_path=None):
    plt.figure(figsize=(6, 6))
    
    recalls = np.array(recalls)
    precisions = np.array(precisions)
    
    if method == 'area':
        # Add points (0,0) and (1,0) to close the curve
        x = np.concatenate(([0], recalls, [1]))
        y = np.concatenate(([0], precisions, [0]))
        
        # Fill Area under Curve zone
        plt.fill_between(x, y, alpha=0.3, color='lightblue', label='AUC')
        
        # # Compute Area
        # area = np.sum((x[1:] - x[:-1]) * y[1:])
        
    else:  # method == 'interp'
        interp_points = np.arange(0.0, 1.1, 0.1)
        interp_precisions = []
        
        for t in interp_points:
            if np.sum(recalls >= t) == 0:
                p = 0
            else:
                p = np.max(precisions[recalls >= t])
            interp_precisions.append(p)
        
        # Fill Area under Curve zone
        plt.fill_between(interp_points, interp_precisions, alpha=0.3, color='lightblue', label='AUC')
        
    # Plot the curve
    plt.plot(recalls, precisions, 'b-', linewidth=2, label='PR curve')
    plt.scatter(recalls, precisions, color='red', s=30, label='Points (P,R)')

    # Plot area value
    plt.text(0.3, 0.3, f'AP = {ap:.3f}', transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision/Recall Curve - {label_name}')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(0.7, 0.5), loc='center left', borderaxespad=0.)    
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    
    # Save figure
    if not save_path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"/home/hillelme/DL_project/FasterRCNN-PyTorch/results/graphs/PR_Curve_{timestamp}.png"
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()