import argparse
from utils import init_dir
from earl_trainer import EARLTrainer


# Function to set random seed for reproducibility
def set_seed(seed):
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# Function to experiment with multiple hyperparameters
def experiment(args):
    # Set random seed for reproducibility
    set_seed(args.seed)

    # Initialize EARLTrainer with the current hyperparameters
    trainer = EARLTrainer(args)

    # Train the model
    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Task-level settings
    parser.add_argument('--data_path', default='./data/FB15k-237')
    parser.add_argument('--task_name', default='rotate_fb15k237_dim150_finalopentest')

    # File settings
    parser.add_argument('--state_dir', default='./state', type=str)
    parser.add_argument('--log_dir', default='./log', type=str)
    parser.add_argument('--tb_log_dir', default='./tb_log', type=str)

    # Training settings
    parser.add_argument('--num_step', default=100000, type=int)
    parser.add_argument('--train_bs', default=1024, type=int)
    parser.add_argument('--eval_bs', default=16, type=int)
    parser.add_argument('--num_neg', default=256, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--log_per_step', default=10, type=int)
    parser.add_argument('--check_per_step', default=100, type=int)
    parser.add_argument('--early_stop_patience', default=20, type=int)
    parser.add_argument('--top_k', default=10, type=int)
    parser.add_argument('--res_ent_ratio', default='0p1', type=str)

    # Model settings
    parser.add_argument('--adv_temp', default=1, type=float)
    parser.add_argument('--gamma', default=10, type=float)
    parser.add_argument('--dim', default=150, type=int)
    parser.add_argument('--num_layers', default=2, type=int)
    parser.add_argument('--num_rel', default=None)
    parser.add_argument('--num_ent', default=None)

    # Device settings
    parser.add_argument('--gpu', default='cuda:0', type=str)
    parser.add_argument('--cpu_num', default=10, type=int)
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility")

    args = parser.parse_args()
    init_dir(args)

    # Experimentation: Loop through different values of hyperparameters
    dim_values = [50, 100, 150, 200]  # Example embedding dimensions
    gamma_values = [1.0, 2.0, 10.0]  # Example gamma (margin) values
    lr_values = [0.001, 0.01, 0.1]  # Example learning rates
    batch_size_values = [256, 512, 1024]  # Example batch sizes

    # Loop over all combinations of hyperparameters
    for dim in dim_values:
        for gamma in gamma_values:
            for lr in lr_values:
                for batch_size in batch_size_values:
                    # Update args with the current combination of hyperparameters
                    args.dim = dim
                    args.gamma = gamma
                    args.lr = lr
                    args.train_bs = batch_size  # Update batch size

                    # Print current experiment configuration
                    print(f"Running experiment with dim={dim}, gamma={gamma}, lr={lr}, batch_size={batch_size}")

                    # Update the embedding dimensions for RotatE model
                    args.ent_dim = args.dim * 2
                    args.rel_dim = args.dim

                    # Run the experiment
                    experiment(args)


