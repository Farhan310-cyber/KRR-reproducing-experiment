import os
import json
import argparse
import copy
from collections import OrderedDict

def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)

def load_default_parameters():
    """Load default model parameters for testing."""
    
    # Default model parameters
    params = {
        # Model weakness parameters
        "weakness": {
            "baseline_scores": {
                "Long-tail entities": 0.6,
                "Complex relations": 0.5,
                "Multi-hop reasoning": 0.7,
                "Symmetric relations": 0.4,
                "Antisymmetric relations": 0.5,
                "Transitive relations": 0.6,
                "One-to-many relations": 0.5,
                "Many-to-many relations": 0.6,
                "Compositional patterns": 0.7
            },
            "model_modifiers": {
                "EARL": {
                    "Long-tail entities": -0.2,
                    "Complex relations": -0.15,
                    "Multi-hop reasoning": -0.1,
                    "Symmetric relations": -0.05,
                    "Antisymmetric relations": -0.1,
                    "Transitive relations": -0.1,
                    "One-to-many relations": -0.15,
                    "Many-to-many relations": -0.1,
                    "Compositional patterns": -0.15
                },
                "RotatE": {
                    "Long-tail entities": -0.1,
                    "Complex relations": -0.1,
                    "Multi-hop reasoning": 0.0,
                    "Symmetric relations": -0.15,
                    "Antisymmetric relations": -0.2,
                    "Transitive relations": -0.05,
                    "One-to-many relations": -0.05,
                    "Many-to-many relations": 0.0,
                    "Compositional patterns": -0.1
                },
                "TransE": {
                    "Long-tail entities": 0.0,
                    "Complex relations": 0.1,
                    "Multi-hop reasoning": 0.1,
                    "Symmetric relations": 0.1,
                    "Antisymmetric relations": -0.1,
                    "Transitive relations": -0.1,
                    "One-to-many relations": 0.15,
                    "Many-to-many relations": 0.2,
                    "Compositional patterns": 0.0
                },
                "DistMult": {
                    "Long-tail entities": 0.05,
                    "Complex relations": 0.0,
                    "Multi-hop reasoning": 0.1,
                    "Symmetric relations": -0.2,
                    "Antisymmetric relations": 0.2,
                    "Transitive relations": 0.05,
                    "One-to-many relations": 0.0,
                    "Many-to-many relations": 0.05,
                    "Compositional patterns": 0.1
                },
                "ComplEx": {
                    "Long-tail entities": 0.0,
                    "Complex relations": -0.05,
                    "Multi-hop reasoning": 0.05,
                    "Symmetric relations": -0.1,
                    "Antisymmetric relations": -0.1,
                    "Transitive relations": 0.0,
                    "One-to-many relations": -0.05,
                    "Many-to-many relations": -0.05,
                    "Compositional patterns": 0.0
                }
            }
        },
        
        # Hyperparameter sensitivity parameters
        "sensitivity": {
            "baseline_scores": {
                "Learning rate": 0.7,
                "Embedding dimension": 0.5,
                "Batch size": 0.4,
                "Negative samples": 0.6,
                "Regularization": 0.5,
                "Margin": 0.4,
                "Optimizer choice": 0.3
            },
            "model_modifiers": {
                "EARL": {
                    "Learning rate": -0.1,
                    "Embedding dimension": 0.0,
                    "Batch size": -0.05,
                    "Negative samples": -0.1,
                    "Regularization": -0.05,
                    "Margin": 0.0,
                    "Optimizer choice": -0.05
                },
                "RotatE": {
                    "Learning rate": 0.0,
                    "Embedding dimension": 0.1,
                    "Batch size": 0.0,
                    "Negative samples": 0.05,
                    "Regularization": -0.05,
                    "Margin": 0.1,
                    "Optimizer choice": 0.0
                },
                "TransE": {
                    "Learning rate": 0.05,
                    "Embedding dimension": -0.05,
                    "Batch size": -0.05,
                    "Negative samples": 0.0,
                    "Regularization": -0.1,
                    "Margin": 0.15,
                    "Optimizer choice": -0.05
                },
                "DistMult": {
                    "Learning rate": 0.05,
                    "Embedding dimension": 0.0,
                    "Batch size": 0.0,
                    "Negative samples": -0.05,
                    "Regularization": 0.1,
                    "Margin": -0.1,
                    "Optimizer choice": 0.0
                },
                "ComplEx": {
                    "Learning rate": 0.0,
                    "Embedding dimension": 0.05,
                    "Batch size": 0.0,
                    "Negative samples": 0.0,
                    "Regularization": 0.05,
                    "Margin": -0.05,
                    "Optimizer choice": 0.05
                }
            }
        },
        
        # Training efficiency parameters
        "efficiency": {
            "baseline_metrics": {
                "Training time (hours)": 10,
                "Memory usage (GB)": 4,
                "Parameters (millions)": 50,
                "Convergence (epochs)": 100,
                "Throughput (triples/sec)": 5000
            },
            "model_modifiers": {
                "EARL": {
                    "Training time (hours)": 1.2,
                    "Memory usage (GB)": 1.1,
                    "Parameters (millions)": 0.8,
                    "Convergence (epochs)": 0.9,
                    "Throughput (triples/sec)": 0.9
                },
                "RotatE": {
                    "Training time (hours)": 1.1,
                    "Memory usage (GB)": 1.0,
                    "Parameters (millions)": 1.0,
                    "Convergence (epochs)": 1.0,
                    "Throughput (triples/sec)": 0.95
                },
                "TransE": {
                    "Training time (hours)": 0.8,
                    "Memory usage (GB)": 0.9,
                    "Parameters (millions)": 0.9,
                    "Convergence (epochs)": 1.1,
                    "Throughput (triples/sec)": 1.2
                },
                "DistMult": {
                    "Training time (hours)": 0.7,
                    "Memory usage (GB)": 0.8,
                    "Parameters (millions)": 0.8,
                    "Convergence (epochs)": 1.2,
                    "Throughput (triples/sec)": 1.3
                },
                "ComplEx": {
                    "Training time (hours)": 0.9,
                    "Memory usage (GB)": 1.1,
                    "Parameters (millions)": 1.1,
                    "Convergence (epochs)": 1.0,
                    "Throughput (triples/sec)": 1.1
                }
            }
        },
        
        # Unseen data performance parameters
        "unseen_performance": {
            "baseline_performance": {
                "In-domain test": 0.5,
                "Similar domain": 0.4,
                "Cross-domain": 0.3,
                "Temporal shift": 0.35,
                "Entity distribution shift": 0.38
            },
            "model_modifiers": {
                "EARL": {
                    "In-domain test": 1.2,
                    "Similar domain": 1.25,
                    "Cross-domain": 1.3,
                    "Temporal shift": 1.2,
                    "Entity distribution shift": 1.3
                },
                "RotatE": {
                    "In-domain test": 1.1,
                    "Similar domain": 1.1,
                    "Cross-domain": 1.05,
                    "Temporal shift": 1.1,
                    "Entity distribution shift": 1.1
                },
                "TransE": {
                    "In-domain test": 1.0,
                    "Similar domain": 0.95,
                    "Cross-domain": 0.9,
                    "Temporal shift": 0.95,
                    "Entity distribution shift": 0.9
                },
                "DistMult": {
                    "In-domain test": 0.95,
                    "Similar domain": 0.9,
                    "Cross-domain": 0.85,
                    "Temporal shift": 0.9,
                    "Entity distribution shift": 0.85
                },
                "ComplEx": {
                    "In-domain test": 1.05,
                    "Similar domain": 1.0,
                    "Cross-domain": 0.95,
                    "Temporal shift": 1.0,
                    "Entity distribution shift": 0.95
                }
            }
        }
    }
    
    return params

def save_parameters(params, output_file):
    """Save parameters to JSON file."""
    with open(output_file, 'w') as f:
        json.dump(params, f, indent=4)

def load_parameters(input_file):
    """Load parameters from JSON file."""
    with open(input_file, 'r') as f:
        return json.load(f)

def create_test_scenarios():
    """Create different test scenarios by modifying the default parameters."""
    default_params = load_default_parameters()
    
    # Create directory for test scenarios
    ensure_dir("./test_scenarios")
    
    # Scenario 1: Improve EARL's performance on multi-hop reasoning
    scenario1 = copy.deepcopy(default_params)
    scenario1["weakness"]["model_modifiers"]["EARL"]["Multi-hop reasoning"] = -0.3  # Better performance
    save_parameters(scenario1, "./test_scenarios/improved_earl_multihop.json")
    
    # Scenario 2: Make TransE better at handling many-to-many relations
    scenario2 = copy.deepcopy(default_params)
    scenario2["weakness"]["model_modifiers"]["TransE"]["Many-to-many relations"] = -0.1  # Better performance
    save_parameters(scenario2, "./test_scenarios/improved_transe_manytomany.json")
    
    # Scenario 3: Reduce RotatE's sensitivity to embedding dimension
    scenario3 = copy.deepcopy(default_params)
    scenario3["sensitivity"]["model_modifiers"]["RotatE"]["Embedding dimension"] = -0.1  # Less sensitive
    save_parameters(scenario3, "./test_scenarios/reduced_rotate_sensitivity.json")
    
    # Scenario 4: Make EARL faster but with more parameters
    scenario4 = copy.deepcopy(default_params)
    scenario4["efficiency"]["model_modifiers"]["EARL"]["Training time (hours)"] = 0.9  # Faster
    scenario4["efficiency"]["model_modifiers"]["EARL"]["Parameters (millions)"] = 1.2  # More parameters
    save_parameters(scenario4, "./test_scenarios/faster_earl.json")
    
    # Scenario 5: Improve all models' cross-domain performance
    scenario5 = copy.deepcopy(default_params)
    for model in scenario5["unseen_performance"]["model_modifiers"]:
        scenario5["unseen_performance"]["model_modifiers"][model]["Cross-domain"] += 0.1  # Better cross-domain
    save_parameters(scenario5, "./test_scenarios/improved_crossdomain.json")
    
    # Scenario 6: Compare different embedding dimensions
    # Create scenarios with different embedding dimensions (50, 100, 200, 300, 500)
    dims = [50, 100, 200, 300, 500]
    for dim in dims:
        scenario_dim = copy.deepcopy(default_params)
        # Adjust training time and parameters based on dimension
        factor = dim / 150  # 150 is the default dimension in the original analysis
        
        for model in scenario_dim["efficiency"]["model_modifiers"]:
            # Training time increases with dimension
            scenario_dim["efficiency"]["model_modifiers"][model]["Training time (hours)"] *= factor
            # Parameters increase with dimension
            scenario_dim["efficiency"]["model_modifiers"][model]["Parameters (millions)"] *= factor
            # Performance might improve with larger dimensions
            for dataset in scenario_dim["unseen_performance"]["model_modifiers"][model]:
                # Small improvement with larger dimensions (diminishing returns)
                scenario_dim["unseen_performance"]["model_modifiers"][model][dataset] *= (1 + 0.1 * (factor - 1))
        
        save_parameters(scenario_dim, f"./test_scenarios/dimension_{dim}.json")
    
    print("Created test scenarios in ./test_scenarios/")

def run_analysis_with_parameters(params_file, output_dir):
    """Run the model analysis with custom parameters."""
    # Import the analysis module
    import analyze_model_characteristics as analyzer
    
    # Load parameters
    params = load_parameters(params_file)
    
    # Create output directory
    ensure_dir(output_dir)
    ensure_dir(os.path.join(output_dir, 'visualizations'))
    
    # Generate data with custom parameters
    weakness_scores = analyzer.generate_model_weaknesses(
        baseline_scores=params["weakness"]["baseline_scores"],
        model_modifiers=params["weakness"]["model_modifiers"]
    )
    
    sensitivity_scores = analyzer.generate_hyperparameter_sensitivity(
        baseline_scores=params["sensitivity"]["baseline_scores"],
        model_modifiers=params["sensitivity"]["model_modifiers"]
    )
    
    efficiency_metrics = analyzer.generate_training_efficiency(
        baseline_metrics=params["efficiency"]["baseline_metrics"],
        model_modifiers=params["efficiency"]["model_modifiers"]
    )
    
    unseen_performance = analyzer.generate_unseen_data_performance(
        baseline_performance=params["unseen_performance"]["baseline_performance"],
        model_modifiers=params["unseen_performance"]["model_modifiers"]
    )
    
    # Generate visualizations
    analyzer.plot_model_weaknesses(
        weakness_scores,
        os.path.join(output_dir, 'visualizations', 'model_weaknesses.png')
    )
    
    analyzer.plot_hyperparameter_sensitivity(
        sensitivity_scores,
        os.path.join(output_dir, 'visualizations', 'hyperparameter_sensitivity.png')
    )
    
    analyzer.plot_training_efficiency(
        efficiency_metrics,
        os.path.join(output_dir, 'visualizations', 'training_efficiency.png')
    )
    
    analyzer.plot_unseen_data_performance(
        unseen_performance,
        os.path.join(output_dir, 'visualizations', 'unseen_data_performance_mrr.png'),
        metric='MRR'
    )
    
    analyzer.plot_unseen_data_performance(
        unseen_performance,
        os.path.join(output_dir, 'visualizations', 'unseen_data_performance_hits10.png'),
        metric='Hits@10'
    )
    
    # Generate CSV files
    analyzer.generate_model_analysis_csv(output_dir, 
                                        weakness_scores, 
                                        sensitivity_scores, 
                                        efficiency_metrics, 
                                        unseen_performance)
    
    print(f"Analysis with custom parameters complete. Results saved to {output_dir}")
    
    # Generate HTML report
    import generate_model_analysis_report as reporter
    report_file = os.path.join(output_dir, "analysis_report.html")
    reporter.generate_html_report(output_dir, report_file)
    print(f"Report generated: {report_file}")

def main():
    parser = argparse.ArgumentParser(description="Customize model parameters for testing")
    parser.add_argument("--action", choices=["create", "export", "import", "run"], default="create",
                        help="Action to perform: create test scenarios, export default parameters, import parameters, or run analysis")
    parser.add_argument("--input_file", help="Input parameter file for import or run")
    parser.add_argument("--output_file", default="./model_parameters.json", 
                        help="Output file for exported parameters")
    parser.add_argument("--output_dir", default="./custom_analysis", 
                        help="Output directory for analysis results")
    
    args = parser.parse_args()
    
    if args.action == "create":
        create_test_scenarios()
    
    elif args.action == "export":
        params = load_default_parameters()
        save_parameters(params, args.output_file)
        print(f"Default parameters exported to {args.output_file}")
    
    elif args.action == "import":
        if not args.input_file:
            print("Error: --input_file is required for import action")
            return
        
        params = load_parameters(args.input_file)
        save_parameters(params, args.output_file)
        print(f"Parameters imported from {args.input_file} and saved to {args.output_file}")
    
    elif args.action == "run":
        if not args.input_file:
            print("Error: --input_file is required for run action")
            return
        
        run_analysis_with_parameters(args.input_file, args.output_dir)

if __name__ == "__main__":
    main() 