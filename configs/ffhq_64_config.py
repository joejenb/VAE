import wandb

wandb.init(project="VAE")
wandb.watch_called = False # Re-run the model without restarting the runtime, unnecessary after our next release

# WandB – Config is a variable that holds and saves hyperparameters and inputs
config = wandb.config          # Initialize config
config.batch_size = 64          # input batch size for training (default: 64)
config.epochs = 100             # number of epochs to train (default: 10)
config.no_cuda = False         # disables CUDA training
config.seed = 42               # random seed (default: 42)
config.image_size = 64
config.log_interval = 1     # how many batches to wait before logging training status
config.learning_rate = 1e-3
config.momentum = 0.1

config.num_hiddens = 128
config.num_residual_layers = 2
config.num_residual_hiddens = 32
config.num_embeddings = 512
config.num_filters = 64
config.embedding_dim = config.num_filters
config.num_channels = 3
config.data_set = "FFHQ"
config.representation_dim = 8
