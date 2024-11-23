# Time-Series Generative Adversarial Network (TimeGAN) Implementation
This project uses a Time-Series Generative Adversarial Network (TimeGAN) for generating synthetic time-series data. TimeGAN combines the strengths of GANs (Generative Adversarial Networks) with the time-series data structure, leveraging the capabilities of deep learning techniques to generate realistic, sequential data.

## Overview

The approach presented here is based on the paper **Time-series Generative Adversarial Networks** by  `Jinsung Yoon`, `Daniel Jarrett`, and `Mihaela van der Schaar (2019)`. This documentation provides an overview of the key components and steps involved in implementing the TimeGAN for time-series data generation.

1. ## Imports and Initial Setup
The first step in this implementation is importing necessary libraries for data processing, machine learning, and visualization.

- `numpy` and `pandas`: Used for numerical and data manipulation tasks, respectively. `pandas` is especially helpful for reading and processing the dataset, while `numpy` is used for mathematical operations.
- `pyarrow.parquet`: This module is used to read and write Parquet files, which is a columnar storage file format.
- `tensorflow`: This library is used for building and training the TimeGAN model. It contains functions for constructing neural networks and optimizers.
- `matplotlib` and `seaborn`: These libraries are used for plotting and visualizing the dataset, allowing us to understand the data distribution and detect outliers visually.
- `tqdm`: A progress bar library, useful for tracking the progress of loops during data processing or model training.
- `sklearn`: Scikit-learn is used for additional data processing and machine learning utilities like scaling and dimensionality reduction.

2. ## Environment Setup
The script checks for available GPU devices and sets memory growth for TensorFlow, which is an important step in ensuring that the GPU is used efficiently without running into memory issues. If there is no GPU available, it defaults to using the CPU.

3. ##  Directory Setup for Experiment Logging
This section sets up a directory structure to log the results of the experiments, ensuring that files are saved properly during the model’s execution. It uses `Path` from the `pathlib` module to create a directory where the results of each experiment will be stored.

4. ## Data Loading and Visualization
- **Loading Data**: The data is loaded from two CSV files: `long_format.csv`(non-interpolated) or `timegan_long_format.csv`(interpolated). Both datasets contain biomarker data for various subjects over time. The `subject_id` and `time_hours` columns are dropped, leaving only the relevant biomarker time-series data for model training.

    - The non-interpolated dataset contains raw time-series data with irregular time intervals.
    - The interpolated dataset contains the same data with missing time points filled in to create regular time intervals, ensuring uniformity for training the model.

- **Visualizing Data** : Before training, it is important that we  understand the data distribution. The code uses `seaborn` to plot kernel density estimates (KDEs) and histograms for two biomarkers, `GFAP` and `UCHL1`. This helps visualize the distribution of values for these biomarkers, allowing us to detect skewness, outliers, or patterns.

5. ## Outlier Detection and Data Cleaning
Outliers can significantly affect the training of the model. In this implementation, **z-scores** are used to identify outlier in the `GFAP ` and `UCHL1` columns. Any values with a z-score greater than 2 or less than -2 are considered outliers. These outliers are then removed from the dataset.

Once identified, the outliers are removed by filtering the data for subjects that do not contain outliers in either `GFAP` or `UCHL1`

6. ## Data Normalization
Normalization is very important for deep learning models as it helps prevent certain features from dominating the learning process. In this implementation, **MinMaxScaler** from sklearn is used to scale the data to a range between 0 and 1.

7. ## Time-Series Window Creation
TimeGAN works with time-series data and requires data to be split into sliding windows. In this implementation, a rolling window of length 6 is created, which means each sequence of 6 time steps will be used to predict the next time step.

Each window is used as a separate sample for training, and the data is converted into a TensorFlow dataset using the `tf.data.Dataset` API, which efficiently handles data batching and shuffling for training.

8. ## Random Series Generator
To create synthetic data for the generator in the GAN architecture, random series are generated. This series is produced from a uniform distribution and has the same shape as the real data.

essentially, this generator provides random sequences that are used as input for the generator network of the TimeGAN during training. This is the starting point for generating synthetic data that the model will learn to improve.

## Next Steps
With the data preprocessed and cleaned, the next step in the implementation is to define the architecture of TimeGAN and begin training using this prepared data. This includes setting up the TimeGAN components, adversarial training, and supervising the generation of synthetic data.

# TimeGAN Components Documentation

The TimeGAN model consists of several components, each designed to perform a specific task in generating synthetic time-series data. Here's a breakdown of the components used in the provided code:

## 1. Network Parameters
The network parameters define key aspects of the model architecture and training configuration. These parameters significantly influence the model’s performance and outcomes by determining the complexity of the neural networks, the capacity for learning temporal patterns, and the duration of training. For instance, the number of hidden dimensions affects how well the model captures underlying patterns, while the number of layers influences its ability to model hierarchical structures in time-series data. The training steps dictate how long the model trains to minimize loss, and the gamma parameter helps balance different objectives during training.

- **`hidden_dim = 24`**: This sets the number of units in the hidden layers for each RNN (Recurrent Neural Network) component.
- **`num_layers = 3`**: Specifies the number of layers in the RNN architecture.
- **`train_steps = 1000`**: Defines the number of training steps for the model.
- **`gamma = 1`**: A hyperparameter used for tuning model behavior (e.g., during training loss calculation).

## 2. RNN Cell Definition
To create the necessary components for the TimeGAN model, we use an RNN-based architecture. The `make_rnn` function defines a basic recurrent neural network (RNN) cell using the GRU (Gated Recurrent Unit) architecture. The function allows for flexibility in the number of layers and the number of units in each layer. Here's what the function does:

### Input:
- **`n_layers`**: Number of layers in the RNN.
- **`hidden_units`**: Number of hidden units (neurons) in each layer.
- **`output_units`**: Number of output units.
- **`name`**: The name of the model being constructed (e.g., Embedder, Generator).

### Output:
A Sequential Keras model that consists of a series of GRU layers followed by Dense layers with ReLU activations and dropout for regularization. ReLU activations help introduce non-linearity to the model, enabling it to capture complex relationships in the data, while dropout prevents overfitting by randomly deactivating a fraction of the neurons during training. The final layer uses a sigmoid activation function for producing output between 0 and 1.

## 3. Model Components
The TimeGAN model consists of several key components, each serving a distinct function:

### Embedder
A sequence model that transforms input data into an embedded space. It uses an RNN architecture with GRU units and outputs data of the same dimension as the input.

### Recovery
This model attempts to reverse the embedding transformation and reconstruct the original time-series data.

### Generator
This model generates synthetic time-series data. It takes the output of the latent space (random noise) and learns to generate data that resembles the real data distribution.

### Discriminator
The discriminator is a binary classifier that distinguishes between real and generated data. It plays a critical role in the adversarial learning process.

### Supervisor
This model supervises the generator during training by guiding it toward generating realistic data through adversarial loss.

Each of these models is based on an RNN (specifically GRU) architecture and has been designed to handle time-series data.

## 4. Loss Functions

### Mean Squared Error (MSE)
Used to measure the error between predicted and true values, especially for the generator’s reconstruction process.

### Binary Cross-Entropy (BCE)
Used by the discriminator to differentiate between real and fake data. The BCE loss function penalizes the discriminator for misclassifying fake data as real or vice versa.

### Generator Loss
The generator is trained to produce data that resembles real data. The loss is calculated based on the difference in statistical moments (mean and variance) between the generated and real data. The `get_generator_moment_loss` function computes this:

- **Mean loss**: The absolute difference between the mean of real and generated data.
- **Variance loss**: The absolute difference between the variance of real and generated data.

## 5. Data Preprocessing
Before training the TimeGAN, the data must be preprocessed. The `get_real_data` function loads and uses the prepared data from the subsequent steps for the model:

## 6. Model for Classification
A simple LSTM (Long Short-Term Memory) model is defined in the `get_model` function to perform classification tasks on the time-series data. This task complements the TimeGAN model by providing a way to evaluate the quality of the generated time-series data, ensuring that it is realistic and preserves meaningful patterns for downstream analysis. Here's what this model does:

### Input:
Time-series sequences of length `seq_len-1` and `n_seq` features (biomarkers).

### Output:
A classification output with `n_seq` units, corresponding to the number of biomarkers.

The model is compiled using the Adam optimizer and the Mean Absolute Error (MAE) loss function.


## Next Steps 
the next step will include model training


# Model Training 

The model trains in multiple phases: first, it trains the autoencoder and supervisor networks separately, then it jointly trains the generator, discriminator, and embedder. The final goal is to generate synthetic data that closely mirrors the characteristics of the original time-series data.

## 2. Parameter Initialization:
The script starts by defining several hyperparameters for training:

- **Hidden Dimension**: The size of the hidden layers in the model.
- **Number of Layers**: The number of layers in the networks.
- **Epochs**: The number of training steps the model will undergo.
- **Sequence Length**: The length of each time-series sequence.
- **Batch Size**: How many samples will be processed at once.

These parameters are stored in a JSON file for reproducibility and to easily reference the experimental setup.

## 3. Data Input Structure:
The model takes two types of input:

- **Real Data (X)**: Actual time-series data that the model will learn to replicate.
- **Random Noise (Z)**: Used by the generator to create fake data. This noise serves as the starting point for the synthetic data generation.

These inputs are placeholders for the real and random data during model training.

## 4. Autoencoder Training:
The autoencoder in TimeGAN is responsible for learning the latent space representation of the real time-series data. The embedder compresses the data into a lower-dimensional space (latent space), and the recovery layer reconstructs the data from that compressed representation.

The autoencoder is trained by comparing the original time-series data to the reconstructed data, measuring the difference using a loss function (in this case, Mean Squared Error, MSE). The goal is for the autoencoder to minimize this reconstruction error. The optimizer updates the weights of the autoencoder based on this loss.

## 5. Supervisor Training:
Once the autoencoder is trained, the next step is to train the supervisor, which predicts the next sequence in the time series. Given the latent representation of the real data, the supervisor predicts what the next time step should be. The loss function compares the predicted next step to the actual next step in the sequence.

Training the supervisor is important because it allows the generator to learn the temporal dependencies in time-series data. The generator must understand how the past data (in the latent space) relates to future data.

## 6. Generator and Discriminator Training:
After the supervisor is trained, the model enters its adversarial phase, where the generator and discriminator are trained together in a GAN setup:

- The generator learns to produce fake time-series data from random noise (Z). The generator tries to make this fake data as realistic as possible.
- The discriminator is trained to distinguish between real data (X) and fake data (Z). It outputs a probability indicating whether the input data is real or generated.

During training, the generator receives feedback from the discriminator. If the discriminator correctly classifies the data as fake, the generator is penalized. The generator’s goal is to fool the discriminator into thinking the fake data is real, while the discriminator works to correctly identify real and fake data.

The generator is trained more frequently than the discriminator, and the losses are calculated for different components:

- **Unsupervised Loss**: This is the binary cross-entropy (BCE) loss that comes from the adversarial setup, where the generator attempts to fool the discriminator.
- **Supervised Loss**: This loss measures how well the generator is able to predict the next time step (in the same manner as the supervisor).
- **Moment Loss**: This custom loss ensures that the statistical properties (moments) of the generated data match those of the real data.

## 7. Joint Training of Generator, Discriminator, and Embedder:
In this phase, the generator, discriminator, and embedder are trained together. The joint training loop optimizes all three components simultaneously. The key steps include:

- Training the generator to minimize its adversarial, supervised, and moment losses.
- Training the embedder to improve its representation of the real data.
- Training the discriminator to distinguish between real and fake data.

This joint training is crucial because it ensures that all parts of the model improve together, allowing the generator to produce high-quality synthetic data.

## 8. Model Saving:
After the model finishes training, the trained generator (the model responsible for creating synthetic time-series data) is saved. This allows you to generate synthetic data later by passing random noise into the trained generator.

## 9. Synthetic Data Generation:
Once the model is trained, the generator can take random noise as input and output synthetic time-series data. This generated data should resemble the real data, capturing its temporal patterns and statistical properties.
