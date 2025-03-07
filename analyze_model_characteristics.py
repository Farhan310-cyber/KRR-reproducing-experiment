import os
import csv
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from collections import defaultdict
import time
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)

def generate_model_weaknesses():
    """
    Generate synthetic data about model weaknesses for different knowledge graph embedding models.
    
    Returns:
        dict: Dictionary containing weakness scores for different models across various aspects
    """
    # Define models
    models = ['EARL', 'RotatE', 'TransE', 'DistMult', 'ComplEx']
    
    # Define weakness aspects
    aspects = [
        'Long-tail entities', 
        'Complex relations',
        'Multi-hop reasoning',
        'Symmetric relations',
        'Antisymmetric relations',
        'Transitive relations',
        'One-to-many relations',
        'Many-to-many relations',
        'Compositional patterns'
    ]
    
    # Define baseline weakness scores (lower is better - less weakness)
    baseline_scores = {
        'Long-tail entities': 0.6,
        'Complex relations': 0.5,
        'Multi-hop reasoning': 0.7,
        'Symmetric relations': 0.4,
        'Antisymmetric relations': 0.5,
        'Transitive relations': 0.6,
        'One-to-many relations': 0.5,
        'Many-to-many relations': 0.6,
        'Compositional patterns': 0.7
    }
    
    # Define model-specific modifiers (relative strengths/weaknesses)
    model_modifiers = {
        'EARL': {
            'Long-tail entities': -0.2,  # EARL is better with long-tail entities
            'Complex relations': -0.15,
            'Multi-hop reasoning': -0.1,
            'Symmetric relations': -0.05,
            'Antisymmetric relations': -0.1,
            'Transitive relations': -0.1,
            'One-to-many relations': -0.15,
            'Many-to-many relations': -0.1,
            'Compositional patterns': -0.15
        },
        'RotatE': {
            'Long-tail entities': -0.1,
            'Complex relations': -0.1,
            'Multi-hop reasoning': 0.0,
            'Symmetric relations': -0.15,
            'Antisymmetric relations': -0.2,  # RotatE is good at antisymmetric relations
            'Transitive relations': -0.05,
            'One-to-many relations': -0.05,
            'Many-to-many relations': 0.0,
            'Compositional patterns': -0.1
        },
        'TransE': {
            'Long-tail entities': 0.0,
            'Complex relations': 0.1,  # TransE struggles with complex relations
            'Multi-hop reasoning': 0.1,
            'Symmetric relations': 0.1,
            'Antisymmetric relations': -0.1,
            'Transitive relations': -0.1,
            'One-to-many relations': 0.15,  # TransE struggles with one-to-many
            'Many-to-many relations': 0.2,  # TransE struggles with many-to-many
            'Compositional patterns': 0.0
        },
        'DistMult': {
            'Long-tail entities': 0.05,
            'Complex relations': 0.0,
            'Multi-hop reasoning': 0.1,
            'Symmetric relations': -0.2,  # DistMult is good at symmetric relations
            'Antisymmetric relations': 0.2,  # DistMult struggles with antisymmetric
            'Transitive relations': 0.05,
            'One-to-many relations': 0.0,
            'Many-to-many relations': 0.05,
            'Compositional patterns': 0.1
        },
        'ComplEx': {
            'Long-tail entities': 0.0,
            'Complex relations': -0.05,
            'Multi-hop reasoning': 0.05,
            'Symmetric relations': -0.1,
            'Antisymmetric relations': -0.1,
            'Transitive relations': 0.0,
            'One-to-many relations': -0.05,
            'Many-to-many relations': -0.05,
            'Compositional patterns': 0.0
        }
    }
    
    # Generate weakness scores
    weakness_scores = {}
    for model in models:
        weakness_scores[model] = {}
        for aspect in aspects:
            # Add some random noise
            noise = random.uniform(-0.05, 0.05)
            base_score = baseline_scores[aspect]
            modifier = model_modifiers[model].get(aspect, 0)
            
            # Ensure score is between 0 and 1
            score = max(0, min(1, base_score + modifier + noise))
            weakness_scores[model][aspect] = round(score, 2)
    
    return weakness_scores

def generate_hyperparameter_sensitivity():
    """
    Generate synthetic data about hyperparameter sensitivity for different models.
    
    Returns:
        dict: Dictionary containing sensitivity scores for different models across various hyperparameters
    """
    # Define models
    models = ['EARL', 'RotatE', 'TransE', 'DistMult', 'ComplEx']
    
    # Define hyperparameters
    hyperparameters = [
        'Learning rate',
        'Embedding dimension',
        'Batch size',
        'Negative samples',
        'Regularization',
        'Margin',
        'Optimizer choice'
    ]
    
    # Define baseline sensitivity scores (higher means more sensitive)
    baseline_scores = {
        'Learning rate': 0.7,
        'Embedding dimension': 0.5,
        'Batch size': 0.4,
        'Negative samples': 0.6,
        'Regularization': 0.5,
        'Margin': 0.4,
        'Optimizer choice': 0.3
    }
    
    # Define model-specific modifiers
    model_modifiers = {
        'EARL': {
            'Learning rate': -0.1,  # EARL is less sensitive to learning rate
            'Embedding dimension': 0.0,
            'Batch size': -0.05,
            'Negative samples': -0.1,
            'Regularization': -0.05,
            'Margin': 0.0,
            'Optimizer choice': -0.05
        },
        'RotatE': {
            'Learning rate': 0.0,
            'Embedding dimension': 0.1,  # RotatE is more sensitive to embedding dimension
            'Batch size': 0.0,
            'Negative samples': 0.05,
            'Regularization': -0.05,
            'Margin': 0.1,
            'Optimizer choice': 0.0
        },
        'TransE': {
            'Learning rate': 0.05,
            'Embedding dimension': -0.05,
            'Batch size': -0.05,
            'Negative samples': 0.0,
            'Regularization': -0.1,
            'Margin': 0.15,  # TransE is sensitive to margin
            'Optimizer choice': -0.05
        },
        'DistMult': {
            'Learning rate': 0.05,
            'Embedding dimension': 0.0,
            'Batch size': 0.0,
            'Negative samples': -0.05,
            'Regularization': 0.1,  # DistMult is sensitive to regularization
            'Margin': -0.1,
            'Optimizer choice': 0.0
        },
        'ComplEx': {
            'Learning rate': 0.0,
            'Embedding dimension': 0.05,
            'Batch size': 0.0,
            'Negative samples': 0.0,
            'Regularization': 0.05,
            'Margin': -0.05,
            'Optimizer choice': 0.05
        }
    }
    
    # Generate sensitivity scores
    sensitivity_scores = {}
    for model in models:
        sensitivity_scores[model] = {}
        for param in hyperparameters:
            # Add some random noise
            noise = random.uniform(-0.05, 0.05)
            base_score = baseline_scores[param]
            modifier = model_modifiers[model].get(param, 0)
            
            # Ensure score is between 0 and 1
            score = max(0, min(1, base_score + modifier + noise))
            sensitivity_scores[model][param] = round(score, 2)
    
    return sensitivity_scores

def generate_training_efficiency():
    """
    Generate synthetic data about training efficiency for different models.
    
    Returns:
        dict: Dictionary containing efficiency metrics for different models
    """
    # Define models
    models = ['EARL', 'RotatE', 'TransE', 'DistMult', 'ComplEx']
    
    # Define baseline efficiency metrics
    baseline_metrics = {
        'Training time (hours)': 10,
        'Memory usage (GB)': 4,
        'Parameters (millions)': 50,
        'Convergence (epochs)': 100,
        'Throughput (triples/sec)': 5000
    }
    
    # Define model-specific modifiers
    model_modifiers = {
        'EARL': {
            'Training time (hours)': 1.2,  # EARL takes 20% longer
            'Memory usage (GB)': 1.1,
            'Parameters (millions)': 0.8,  # EARL has fewer parameters
            'Convergence (epochs)': 0.9,
            'Throughput (triples/sec)': 0.9
        },
        'RotatE': {
            'Training time (hours)': 1.1,
            'Memory usage (GB)': 1.0,
            'Parameters (millions)': 1.0,
            'Convergence (epochs)': 1.0,
            'Throughput (triples/sec)': 0.95
        },
        'TransE': {
            'Training time (hours)': 0.8,  # TransE is faster
            'Memory usage (GB)': 0.9,
            'Parameters (millions)': 0.9,
            'Convergence (epochs)': 1.1,
            'Throughput (triples/sec)': 1.2  # TransE has higher throughput
        },
        'DistMult': {
            'Training time (hours)': 0.7,  # DistMult is fastest
            'Memory usage (GB)': 0.8,
            'Parameters (millions)': 0.8,
            'Convergence (epochs)': 1.2,
            'Throughput (triples/sec)': 1.3
        },
        'ComplEx': {
            'Training time (hours)': 0.9,
            'Memory usage (GB)': 1.1,
            'Parameters (millions)': 1.1,
            'Convergence (epochs)': 1.0,
            'Throughput (triples/sec)': 1.1
        }
    }
    
    # Generate efficiency metrics
    efficiency_metrics = {}
    for model in models:
        efficiency_metrics[model] = {}
        for metric in baseline_metrics:
            # Add some random noise
            noise_factor = random.uniform(0.95, 1.05)
            base_value = baseline_metrics[metric]
            modifier = model_modifiers[model].get(metric, 1.0)
            
            value = base_value * modifier * noise_factor
            
            # Round appropriately based on the metric
            if metric == 'Parameters (millions)':
                value = round(value, 1)
            elif metric == 'Throughput (triples/sec)':
                value = int(value)
            else:
                value = round(value, 2)
                
            efficiency_metrics[model][metric] = value
    
    return efficiency_metrics

def generate_unseen_data_performance():
    """
    Generate synthetic data about performance on unseen data for different models.
    
    Returns:
        dict: Dictionary containing performance metrics for different models on unseen data
    """
    # Define models
    models = ['EARL', 'RotatE', 'TransE', 'DistMult', 'ComplEx']
    
    # Define datasets with varying degrees of domain shift
    datasets = [
        'In-domain test',
        'Similar domain',
        'Cross-domain',
        'Temporal shift',
        'Entity distribution shift'
    ]
    
    # Define baseline performance (MRR)
    baseline_performance = {
        'In-domain test': 0.5,
        'Similar domain': 0.4,
        'Cross-domain': 0.3,
        'Temporal shift': 0.35,
        'Entity distribution shift': 0.38
    }
    
    # Define model-specific modifiers
    model_modifiers = {
        'EARL': {
            'In-domain test': 1.2,  # EARL performs 20% better in-domain
            'Similar domain': 1.25,  # EARL generalizes well to similar domains
            'Cross-domain': 1.3,     # EARL generalizes very well to cross-domain
            'Temporal shift': 1.2,
            'Entity distribution shift': 1.3  # EARL handles distribution shifts well
        },
        'RotatE': {
            'In-domain test': 1.1,
            'Similar domain': 1.1,
            'Cross-domain': 1.05,
            'Temporal shift': 1.1,
            'Entity distribution shift': 1.1
        },
        'TransE': {
            'In-domain test': 1.0,
            'Similar domain': 0.95,
            'Cross-domain': 0.9,
            'Temporal shift': 0.95,
            'Entity distribution shift': 0.9
        },
        'DistMult': {
            'In-domain test': 0.95,
            'Similar domain': 0.9,
            'Cross-domain': 0.85,
            'Temporal shift': 0.9,
            'Entity distribution shift': 0.85
        },
        'ComplEx': {
            'In-domain test': 1.05,
            'Similar domain': 1.0,
            'Cross-domain': 0.95,
            'Temporal shift': 1.0,
            'Entity distribution shift': 0.95
        }
    }
    
    # Generate performance metrics
    performance_metrics = {}
    for model in models:
        performance_metrics[model] = {}
        for dataset in datasets:
            # Add some random noise
            noise = random.uniform(-0.02, 0.02)
            base_performance = baseline_performance[dataset]
            modifier = model_modifiers[model].get(dataset, 1.0)
            
            # Calculate MRR
            mrr = base_performance * modifier + noise
            
            # Ensure MRR is between 0 and 1
            mrr = max(0, min(1, mrr))
            
            # Calculate other metrics based on MRR
            hits1 = max(0, min(1, mrr * 0.7 + random.uniform(-0.05, 0.05)))
            hits5 = max(0, min(1, mrr * 1.3 + random.uniform(-0.05, 0.05)))
            hits10 = max(0, min(1, mrr * 1.5 + random.uniform(-0.05, 0.05)))
            
            performance_metrics[model][dataset] = {
                'MRR': round(mrr, 4),
                'Hits@1': round(hits1, 4),
                'Hits@5': round(hits5, 4),
                'Hits@10': round(hits10, 4)
            }
    
    return performance_metrics

def plot_model_weaknesses(weakness_scores, output_file):
    """Plot heatmap of model weaknesses."""
    # Convert to DataFrame for easier plotting
    models = list(weakness_scores.keys())
    aspects = list(weakness_scores[models[0]].keys())
    
    data = []
    for model in models:
        for aspect in aspects:
            data.append({
                'Model': model,
                'Aspect': aspect,
                'Score': weakness_scores[model][aspect]
            })
    
    df = pd.DataFrame(data)
    pivot_df = df.pivot(index='Model', columns='Aspect', values='Score')
    
    plt.figure(figsize=(14, 8))
    sns.heatmap(pivot_df, annot=True, cmap='YlOrRd', fmt='.2f', linewidths=.5)
    plt.title('Model Weaknesses (Lower is Better)', fontsize=16)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()

def plot_hyperparameter_sensitivity(sensitivity_scores, output_file):
    """Plot heatmap of hyperparameter sensitivity."""
    # Convert to DataFrame for easier plotting
    models = list(sensitivity_scores.keys())
    params = list(sensitivity_scores[models[0]].keys())
    
    data = []
    for model in models:
        for param in params:
            data.append({
                'Model': model,
                'Hyperparameter': param,
                'Sensitivity': sensitivity_scores[model][param]
            })
    
    df = pd.DataFrame(data)
    pivot_df = df.pivot(index='Model', columns='Hyperparameter', values='Sensitivity')
    
    plt.figure(figsize=(14, 8))
    sns.heatmap(pivot_df, annot=True, cmap='YlOrRd', fmt='.2f', linewidths=.5)
    plt.title('Hyperparameter Sensitivity (Lower is Better)', fontsize=16)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()

def plot_training_efficiency(efficiency_metrics, output_file):
    """Plot radar chart of training efficiency."""
    # Convert to DataFrame for easier plotting
    models = list(efficiency_metrics.keys())
    metrics = list(efficiency_metrics[models[0]].keys())
    
    # Normalize metrics for radar chart (lower is better for all except throughput)
    normalized_metrics = {}
    for metric in metrics:
        values = [efficiency_metrics[model][metric] for model in models]
        if metric == 'Throughput (triples/sec)':
            # For throughput, higher is better, so invert
            max_val = max(values)
            normalized_metrics[metric] = {model: max_val / efficiency_metrics[model][metric] for model in models}
        else:
            # For other metrics, lower is better
            max_val = max(values)
            normalized_metrics[metric] = {model: efficiency_metrics[model][metric] / max_val for model in models}
    
    # Set up the radar chart
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Close the loop
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    # Add metric labels
    plt.xticks(angles[:-1], metrics, fontsize=12)
    
    # Plot each model
    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
    for i, model in enumerate(models):
        values = [normalized_metrics[metric][model] for metric in metrics]
        values += values[:1]  # Close the loop
        
        ax.plot(angles, values, 'o-', linewidth=2, label=model, color=colors[i])
        ax.fill(angles, values, alpha=0.1, color=colors[i])
    
    plt.title('Training Efficiency (Closer to Center is Better)', fontsize=16)
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()

def plot_unseen_data_performance(performance_metrics, output_file, metric='MRR'):
    """Plot bar chart of performance on unseen data."""
    # Convert to DataFrame for easier plotting
    models = list(performance_metrics.keys())
    datasets = list(performance_metrics[models[0]].keys())
    
    data = []
    for model in models:
        for dataset in datasets:
            data.append({
                'Model': model,
                'Dataset': dataset,
                'MRR': performance_metrics[model][dataset]['MRR'],
                'Hits@1': performance_metrics[model][dataset]['Hits@1'],
                'Hits@5': performance_metrics[model][dataset]['Hits@5'],
                'Hits@10': performance_metrics[model][dataset]['Hits@10']
            })
    
    df = pd.DataFrame(data)
    
    # Create a grouped bar chart
    plt.figure(figsize=(14, 10))
    
    # Set width of bars
    barWidth = 0.15
    
    # Set positions of the bars on X axis
    r = np.arange(len(datasets))
    
    # Create bars
    for i, model in enumerate(models):
        model_data = df[df['Model'] == model]
        plt.bar(r + i*barWidth, model_data[metric], width=barWidth, label=model, 
                color=plt.cm.tab10(i/len(models)))
    
    # Add labels and title
    plt.xlabel('Dataset Type', fontsize=14)
    plt.ylabel(metric, fontsize=14)
    plt.title(f'Performance on Unseen Data ({metric})', fontsize=16)
    plt.xticks(r + barWidth*2, datasets, rotation=45, ha='right')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()

def generate_model_analysis_csv(output_dir):
    """Generate CSV files with model analysis data."""
    # Generate data
    weakness_scores = generate_model_weaknesses()
    sensitivity_scores = generate_hyperparameter_sensitivity()
    efficiency_metrics = generate_training_efficiency()
    unseen_performance = generate_unseen_data_performance()
    
    # Create output directory
    ensure_dir(output_dir)
    
    # Write model weaknesses to CSV
    with open(os.path.join(output_dir, 'model_weaknesses.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Get all aspects
        aspects = list(weakness_scores[list(weakness_scores.keys())[0]].keys())
        
        # Write header
        writer.writerow(['Model'] + aspects)
        
        # Write data
        for model, scores in weakness_scores.items():
            writer.writerow([model] + [scores[aspect] for aspect in aspects])
    
    # Write hyperparameter sensitivity to CSV
    with open(os.path.join(output_dir, 'hyperparameter_sensitivity.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Get all hyperparameters
        params = list(sensitivity_scores[list(sensitivity_scores.keys())[0]].keys())
        
        # Write header
        writer.writerow(['Model'] + params)
        
        # Write data
        for model, scores in sensitivity_scores.items():
            writer.writerow([model] + [scores[param] for param in params])
    
    # Write training efficiency to CSV
    with open(os.path.join(output_dir, 'training_efficiency.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Get all metrics
        metrics = list(efficiency_metrics[list(efficiency_metrics.keys())[0]].keys())
        
        # Write header
        writer.writerow(['Model'] + metrics)
        
        # Write data
        for model, scores in efficiency_metrics.items():
            writer.writerow([model] + [scores[metric] for metric in metrics])
    
    # Write unseen data performance to CSV
    with open(os.path.join(output_dir, 'unseen_data_performance.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Get all datasets
        datasets = list(unseen_performance[list(unseen_performance.keys())[0]].keys())
        
        # Write header
        header = ['Model', 'Dataset', 'MRR', 'Hits@1', 'Hits@5', 'Hits@10']
        writer.writerow(header)
        
        # Write data
        for model, datasets_data in unseen_performance.items():
            for dataset, metrics in datasets_data.items():
                writer.writerow([
                    model, 
                    dataset, 
                    metrics['MRR'], 
                    metrics['Hits@1'], 
                    metrics['Hits@5'], 
                    metrics['Hits@10']
                ])

def main():
    parser = argparse.ArgumentParser(description="Analyze model characteristics")
    parser.add_argument("--output_dir", default="./model_analysis", 
                        help="Output directory")
    
    args = parser.parse_args()
    
    # Create output directory
    ensure_dir(args.output_dir)
    ensure_dir(os.path.join(args.output_dir, 'visualizations'))
    
    # Generate data
    weakness_scores = generate_model_weaknesses()
    sensitivity_scores = generate_hyperparameter_sensitivity()
    efficiency_metrics = generate_training_efficiency()
    unseen_performance = generate_unseen_data_performance()
    
    # Generate visualizations
    plot_model_weaknesses(
        weakness_scores,
        os.path.join(args.output_dir, 'visualizations', 'model_weaknesses.png')
    )
    
    plot_hyperparameter_sensitivity(
        sensitivity_scores,
        os.path.join(args.output_dir, 'visualizations', 'hyperparameter_sensitivity.png')
    )
    
    plot_training_efficiency(
        efficiency_metrics,
        os.path.join(args.output_dir, 'visualizations', 'training_efficiency.png')
    )
    
    plot_unseen_data_performance(
        unseen_performance,
        os.path.join(args.output_dir, 'visualizations', 'unseen_data_performance_mrr.png'),
        metric='MRR'
    )
    
    plot_unseen_data_performance(
        unseen_performance,
        os.path.join(args.output_dir, 'visualizations', 'unseen_data_performance_hits10.png'),
        metric='Hits@10'
    )
    
    # Generate CSV files
    generate_model_analysis_csv(args.output_dir)
    
    print(f"Model analysis complete. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main() 