---
title: Deep Learning Algorithms
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

We will cover:

- **Artificial Neural Network (ANN) / Multi-Layer Perceptron (MLP)**
- **Recurrent Neural Networks (RNN)**
- **Long Short-Term Memory (LSTM)**
- **Gated Recurrent Unit (GRU)**
- **Convolutional Neural Networks (CNN)**

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

#### Install Machine Learning Frameworks and Libraries

```python
pip install tensorflow
```

#### Importing Libraries

```python
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU, Conv1D, Flatten, InputLayer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Import the NumPy library for numerical operations
import numpy as np

# Import the Pandas library for data manipulation and analysis
import pandas as pd

# Import Matplotlib for plotting and visualization
import matplotlib.pyplot as plt

# Import r2_score from scikit-learn for evaluating model performance
from sklearn.metrics import r2_score

# Import StandardScaler from scikit-learn for feature scaling
from sklearn.preprocessing import StandardScaler
```

#### Load the data from the csv file
```python
# Load the data from the csv file
data = pd.read_csv('gdp_data.csv', index_col='Year')
data.head()
```

#### Plot the timeseries of the data
```python
# Create a function to plot the time series
def plot_time_series(column):
    plt.figure(figsize=(10, 6))
    data[column].plot()
    plt.title(f'{column}')
    plt.xlabel('Date')
    plt.ylabel(column)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Create a dropdown widget for selecting the column
column_selector = widgets.Dropdown(
    options=data.columns,
    description='Column:',
    disabled=False,
)

# Link the dropdown widget to the plot_time_series function
interactive_plot = widgets.interactive_output(plot_time_series, {'column': column_selector})

# Display the widget and the interactive plot
display(column_selector, interactive_plot)
```

#### Convert data to numpy

```python
X_train_np = X_train.values
X_test_np = X_test.values
y_train_np = y_train.values
y_test_np = y_test.values
```

#### Normalization 
```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_np)
X_test_scaled = scaler.transform(X_test_np)
```
#### Convert the data into 3d
For RNN, LSTM, GRU, CNN, we need to provide a 3D input [samples, timesteps, features].

```python
X_train_3d = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_test_3d = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))
```

```python
def evaluate_nn_model(model, X_test, y_test, name="NN Model"):
    preds = model.predict(X_test).flatten()
    mse = tf.keras.losses.MeanSquaredError()(y_test, preds).numpy()
    rmse = mse**0.5
    r2 = 1 - ( ( (y_test - preds)**2 ).sum() / ((y_test - y_test.mean())**2).sum() )
    print(f"{name} Performance:")
    print(f"  RMSE: {rmse:,.2f}")
    print(f"  R^2: {r2:.4f}")
    print("-"*40)
```


#### MLP (Multi-Layer Perceptron)
```python
mlp = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dense(64, activation='relu'),
    Dense(1)
])

mlp.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

mlp.fit(X_train_scaled, y_train_np, 
        validation_split=0.2, 
        epochs=50, 
        batch_size=32, 
        callbacks=[early_stop], 
        verbose=0)

evaluate_nn_model(mlp, X_test_scaled, y_test_np, "MLP")
```


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





