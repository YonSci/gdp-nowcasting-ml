---
title: Machine Learning Algorithms
teaching: 130
exercises: 130
questions:
- "How much of the data is missing? Is it a small fraction or a significant portion?"

objectives:
- "Learn the difference between deleting incomplete observations and imputing missing values."


keypoints:
- "Deletion: Simple but risks losing large amounts of data and introducing bias."

---


# Neural Networks and Architectures

## Thought About Neural Networks and Architectures for 7 Seconds

Below is a continuation of the hands-on approach, now focusing on various neural network architectures using the same synthetic dataset for predicting GDP. We will cover:

- **Artificial Neural Network (ANN) / Multi-Layer Perceptron (MLP)**
- **Recurrent Neural Networks (RNN)**
- **Long Short-Term Memory (LSTM)**
- **Gated Recurrent Unit (GRU)**
- **Convolutional Neural Networks (CNN)**

### Note:

- The dataset is not inherently sequential or image-based, which makes the use of RNNs, LSTMs, GRUs, and CNNs less conventional.
- Typically, these models excel with sequential data (RNN/LSTM/GRU) or image data (CNN). Here, we’ll demonstrate the architectures to show how to implement them, but keep in mind that the performance may not improve over simpler models for this tabular data.
- For RNN-based models, we will reshape the data to mimic a sequence dimension of length 1 (just as a demonstration). Similarly, for CNN, we will treat the features as a single "temporal" dimension and apply 1D convolutions.

---

## Brief Introductions

### 1. Artificial Neural Network (ANN) / Multi-Layer Perceptron (MLP)

#### Assumptions:
- No specific assumptions about input distributions; MLPs are universal function approximators.

#### How It Works:
- Composed of layers of artificial neurons. Each neuron computes a weighted sum of inputs and applies a non-linear activation function.

**Pros:**
- Can model complex non-linear relationships.
- Highly flexible and can approximate a wide range of functions.

**Cons:**
- Requires tuning many hyperparameters (number of layers, neurons, etc.).
- May need large amounts of data and careful regularization to avoid overfitting.

---

### 2. Recurrent Neural Network (RNN)

#### Assumptions:
- Designed for sequential data, assumes temporal or sequential correlations.

#### How It Works:
- Processes inputs one step at a time, maintaining a hidden state that captures information about previous steps.

**Pros:**
- Good for sequence data such as time series or textual data.

**Cons:**
- Struggles with long-term dependencies due to vanishing or exploding gradients.
- Not well-suited for non-sequential/tabular data.

---

### 3. Long Short-Term Memory (LSTM)

#### Assumptions:
- Similar to RNN, designed for sequences.

#### How It Works:
- A special kind of RNN that uses gates (input, forget, output) to better capture long-term dependencies.

**Pros:**
- Handles longer sequences better than vanilla RNNs.
- Widely used for time series, speech recognition, language modeling.

**Cons:**
- More complex structure, more parameters.
- Still can be computationally expensive.

---

### 4. Gated Recurrent Unit (GRU)

#### Assumptions:
- Similar to LSTM, also designed for sequences.

#### How It Works:
- A simpler variant of LSTM with fewer gates (update and reset), often providing similar performance.

**Pros:**
- More efficient than LSTM due to fewer parameters.
- Often similar performance to LSTM.

**Cons:**
- Still more complex than vanilla RNN.
- Not always guaranteed to outperform LSTM or simpler methods.

---

### 5. Convolutional Neural Networks (CNN)

#### Assumptions:
- Typically used for image data (2D) or sequential data (1D).
- Assumes local patterns (e.g., spatial or temporal local correlations).

#### How It Works:
- Uses convolutional filters that slide over input data to extract local features.

**Pros:**
- Excellent for image recognition or sequence classification.
- Parameter sharing reduces complexity.

**Cons:**
- Not typically the best choice for plain tabular, non-sequential data.
- Feature engineering might be needed to benefit.





