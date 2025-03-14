# FedResNet_for_CIFAR-10
A federated learning model using ResNet18 for image classification on CIFAR-10 dataset

This is an implementation of a federated learning system for image classification on the CIFAR-10 dataset. The main model is based on ResNet18, and there are also simplified CNN and lightweight SqueezeNet models for different scenarios.

## Key Features
- **FedResNet.py**: Implements a federated learning system using ResNet18. It includes client and server classes, supports multiple client sampling strategies (random, performance-based, and mixed), and applies techniques like FedProx, model compression, and differential privacy.
- **FedCNN.py**: A low - performance version using a simple CNN network. It provides a basic federated learning implementation without advanced techniques like FedProx, compression, and differential privacy.
- **FedSqueezeNet.py**: A lightweight network model suitable for scenarios with a larger number of clients. It supports different optimizers and uses a learning rate scheduler.

## Installation
1. **Create a virtual environment with conda/virtualenv**
    - For conda: `conda create -n fed_learning python=3.8`
    - For virtualenv: `virtualenv -p python3.8 fed_learning`
2. **Activate the virtual environment**
    - For conda: `conda activate fed_learning`
    - For virtualenv: `source fed_learning/bin/activate`
3. **Clone the repo**
    - `git clone <REPO_URL>`
4. **Run**: `cd <PATH_TO_THE_CLONED_REPO>`
5. **Run**: `pip install -r requirements.txt` to install necessary packages.

## Run the code

### FedResNet Results on CIFAR10
1. **Run**: `cd FedResNet`
2. **Run**: `python FedResNet.py --num_clients 50 --num_rounds 20 --local_epochs 3 --batch_size 64 --learning_rate 0.005 --momentum 0.9 --weight_decay 5e-4 --fedprox_mu 0.0001 --dp_noise_scale 0.001 --compression_threshold 0.001 --num_selected_clients 5 --sample_strategy mixed`
    - You can adjust the parameters according to your needs.

### FedCNN Results on CIFAR10
1. **Run**: `cd FedResNet`
2. **Run**: `python FedCNN.py --num_clients 50 --num_rounds 20 --local_epochs 3 --batch_size 64 --learning_rate 0.005`
    - You can adjust the parameters according to your needs.

### FedSqueezeNet Results on CIFAR10
1. **Run**: `cd FedResNet`
2. **Run**: `python FedSqueezeNet.py --num_clients 100 --num_rounds 20 --local_epochs 3 --batch_size 64 --learning_rate 0.005 --optimizer_type SGD --momentum 0.9 --weight_decay 5e-4 --lr_gamma 0.9`
    - You can adjust the parameters according to your needs.

## Explanation of Parameters
- **num_clients**: Number of clients in the federated learning system.
- **num_rounds**: Number of global communication rounds.
- **local_epochs**: Number of local training epochs for each client.
- **batch_size**: Batch size for training.
- **learning_rate**: Learning rate for the optimizer.
- **momentum**: Momentum for the SGD optimizer.
- **weight_decay**: Weight decay for the optimizer.
- **fedprox_mu**: FedProx proximal term coefficient.
- **dp_noise_scale**: Differential privacy noise scale.
- **compression_threshold**: Model update compression threshold.
- **num_selected_clients**: Number of selected clients per round.
- **sample_strategy**: Sampling strategy: random, performance, or mixed.
- **optimizer_type**: Optimizer type: SGD or Adam.
- **lr_gamma**: Learning rate decay factor for the scheduler.

## Output
- **Plots**: The training process will generate a plot showing the test accuracy vs. training rounds, saved in the `./tables` directory.
- **JSON Results**: The training results, including hyperparameters and accuracies, will be saved as a JSON file in the `./res` directory.
