Automated Neural Network Pruning using a Genetic Algorithm
This project provides a complete framework for automatically compressing a Convolutional Neural Network (CNN) using a Genetic Algorithm (GA). The goal is to perform structured channel pruning to find a smaller, more computationally efficient model architecture that maintains high accuracy.

The implementation is written in Python using PyTorch and is designed to be easily configurable and extensible for research purposes.

Features
Structured Pruning: Removes entire channels (filters) from convolutional layers, leading to real-world reductions in model size, memory usage, and computational cost (MACs).

Genetic Algorithm Search: Automates the search for an optimal pruning configuration, balancing model accuracy and efficiency.

Advanced Fitness Evaluation: Incorporates a crucial fine-tuning step during the fitness evaluation, allowing the GA to make more informed decisions by assessing the true potential of a pruned architecture.

Configurable: All key hyperparameters for the dataset, model training, and the GA are centralized in a single Config class for easy experimentation.

Comprehensive Evaluation: Automatically trains a baseline model, runs the pruning evolution, and provides a clear, final comparison of the baseline vs. the pruned model across multiple metrics (Accuracy, Parameters, MACs, Size).

Visualization: Generates a plot of the GA's fitness evolution over generations.

Methodology
The process works in three main stages:

1. Baseline Model
A standard CNN is first trained on the MNIST dataset to establish baseline performance metrics. This model serves as the starting point for the pruning process.

2. Genetic Algorithm for Pruning
A Genetic Algorithm (GA) evolves a population of potential network architectures to find the best one.

Chromosome: A binary vector where each bit corresponds to a channel in the prunable convolutional layers. A 1 means the channel is kept; a 0 means it is removed.

Fitness Function: The quality (fitness) of each chromosome is determined by a weighted score that balances two key objectives:

Validation Accuracy: The performance of the pruned model after a brief fine-tuning session.

Computational Reduction: The percentage of MACs saved compared to the baseline model.

Evolution: The GA uses standard operators—tournament selection, uniform crossover, and bit-flip mutation—to evolve the population over several generations, converging on an optimal solution.

3. Final Model Generation
The best chromosome found by the GA is used to build the final, compressed model. This model is then fully re-trained on the entire training dataset to maximize its accuracy before a final evaluation.

Prerequisites
Ensure you have Python 3.x installed. The required libraries can be installed via pip:

pip install torch torchvision
pip install numpy
pip install matplotlib
pip install thop

How to Run
Save the code as a Python file (e.g., run_pruning.py).

Execute the script from your terminal:

python run_pruning.py

The script will handle everything: downloading the data, training the baseline model (if it doesn't exist), running the GA evolution, and printing the final results.

Configuration
All hyperparameters can be adjusted in the Config class at the top of the script. Key parameters to experiment with include:

Config.TRAINING.BATCH_SIZE: The batch size for training. Reduce this if you encounter memory errors.

Config.TRAINING.FINETUNE_EPOCHS: The number of epochs to fine-tune each candidate during the GA's fitness evaluation.

Config.GA.POP_SIZE & Config.GA.NUM_GENERATIONS: The population size and number of generations for the GA. Increasing these can lead to better results but will take longer.

Config.GA.MUTATION_RATE: The probability of a bit being flipped during mutation.

Config.FITNESS.W_ACC & Config.FITNESS.W_MACS: The weights to control the trade-off between accuracy and MAC reduction in the fitness score.

Expected Output
When you run the script, you will see:

Console Logs: Progress updates for data loading, baseline training, and each generation of the GA evolution.

Final Comparison Table: A formatted table in the console comparing the metrics of the baseline and final pruned models.

Fitness Evolution Plot: A PNG image file named ga_fitness_evolution.png will be saved in the project directory, showing the best and average fitness scores over the generations.

Troubleshooting
Kernel Crash / Memory Error: If the script crashes towards the end, it's likely due to GPU memory exhaustion. The simplest fix is to reduce the BATCH_SIZE in the Config class (e.g., from 128 to 64).
