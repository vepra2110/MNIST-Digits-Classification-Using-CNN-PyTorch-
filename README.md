# MNIST Digits Classification using CNN in PyTorch

A production-ready, highly accurate MNIST digit classifier built using PyTorch. The model uses a Convolutional Neural Network (CNN) with batch normalization, dropout regularization, and learning rate scheduling to achieve a test accuracy of **99.40%** (exceeding the target of >99.2%).

---

## 🛠️ Tech Stack & Dependencies

The project is built using Python 3.11+ and relies on the following core libraries for machine learning, data processing, visualization, and evaluation:

*   **Language:** Python 3.11+
*   **Deep Learning Framework:** [PyTorch](https://pytorch.org/) (`torch`, `torchvision`)
*   **Data Manipulation:** [NumPy](https://numpy.org/)
*   **Visualization & Curves:** [Matplotlib](https://matplotlib.org/)
*   **Evaluation Metrics:** [Scikit-Learn](https://scikit-learn.org/) (for confusion matrix and classification report)

---

## 🚀 Initial Setup

You can set up and run this project using either **uv** (recommended for speed and reproducibility) or standard **pip** with virtual environments.

### Option A: Using `uv` (Recommended)

1.  **Initialize the project:**
    ```bash
    uv init --name mnist_cnn
    ```
2.  **Add dependencies:**
    ```bash
    uv add torch torchvision matplotlib scikit-learn numpy
    ```
3.  **Run the training script:**
    ```bash
    uv run main.py
    ```

### Option B: Using standard `pip`

1.  **Create a virtual environment:**
    ```bash
    python -m venv .venv
    ```
2.  **Activate the virtual environment:**
    *   **Windows (CMD/PowerShell):**
        ```powershell
        .venv\Scripts\activate
        ```
    *   **Linux / macOS:**
        ```bash
        source .venv/bin/activate
        ```
3.  **Install dependencies:**
    ```bash
    pip install torch torchvision matplotlib scikit-learn numpy
    ```
4.  **Run the training script:**
    ```bash
    python main.py
    ```

---

## 🧠 CNN Architecture

The model class `MNISTClassifier` consists of exactly two convolution-pooling feature extraction loops followed by a regularized fully connected network.

```
Input Image [1, 28, 28]
  │
  ├─── Loop 1: Conv2d(1 -> 32, 3x3) ──> BatchNorm2d ──> ReLU ──> MaxPool2d(2x2)
  │            [Shape: 32, 14, 14]
  │
  ├─── Loop 2: Conv2d(32 -> 64, 3x3) ──> BatchNorm2d ──> ReLU ──> MaxPool2d(2x2)
  │            [Shape: 64, 7, 7]
  │
  ├─── Flatten ──> 1D Vector (3136 nodes)
  │
  └─── Fully Connected Layer ──> Linear(3136 -> 128) ──> ReLU ──> Dropout(p=0.5) ──> Linear(128 -> 10)
               [Shape: 10 (Logits)]
```

### Shape Transformations:
*   **Input:** `[batch_size, 1, 28, 28]`
*   **After Conv-Pool Loop 1:** `[batch_size, 32, 14, 14]`
*   **After Conv-Pool Loop 2:** `[batch_size, 64, 7, 7]`
*   **After Flattening:** `[batch_size, 3136]`
*   **Fully Connected 1 (Dense):** `[batch_size, 128]`
*   **Output Logits:** `[batch_size, 10]` *(No Softmax layer is applied to logits during forward pass; numerical stability is handled internally by the loss function)*

---

## ⚙️ Optimization & Scheduling

### Loss Function
*   **`nn.CrossEntropyLoss()`**: Handles computation of log-probabilities and negative log-likelihood loss internally for numerical stability.

### Optimizer
*   **`optim.AdamW`**: Consists of weight decay regularization for model generalization.
    *   **Learning Rate ($lr$):** $1 \times 10^{-3}$
    *   **Weight Decay:** $1 \times 10^{-4}$

### Learning Rate Scheduler
*   **`optim.lr_scheduler.ReduceLROnPlateau`**: Monitors the test loss at the end of each epoch. If the test loss plateaus (fails to improve for a patience of 1 epoch), it decays the learning rate by a factor of $0.5$ (e.g., $1 \times 10^{-3} \rightarrow 5 \times 10^{-4}$). This allows the model to fine-tune its weights as it converges.

---

## 📊 Training Results & Evaluation

The training pipeline was run on CPU for **15 epochs** with a train batch size of 128 and test batch size of 1000. Data augmentation transforms (`RandomRotation(10)` and `RandomAffine(translate=(0.1, 0.1))`) were applied to the training set to prevent overfitting.

### Performance Summary
*   **Final Test Accuracy:** **99.40%** (Exceeds the 99.2% target)
*   **Final Test Loss:** **0.0194**
*   **Average Runtime:** **~35 - 40 minutes** on CPU

### Classification Report (Test Set)
```text
              precision    recall  f1-score   support

           0     0.9990    0.9939    0.9964       980
           1     0.9930    0.9947    0.9938      1135
           2     0.9923    0.9942    0.9932      1032
           3     0.9902    0.9980    0.9941      1010
           4     0.9949    0.9919    0.9934       982
           5     0.9966    0.9910    0.9938       892
           6     0.9948    0.9916    0.9932       958
           7     0.9951    0.9932    0.9942      1028
           8     0.9928    0.9979    0.9954       974
           9     0.9921    0.9931    0.9926      1009

    accuracy                         0.9940     10000
   macro avg     0.9941    0.9940    0.9940     10000
weighted avg     0.9940    0.9940    0.9940     10000
```

### Confusion Matrix
```text
[[ 974    0    3    0    0    0    2    1    0    0]
 [   0 1129    2    1    0    1    1    1    0    0]
 [   1    1 1026    2    0    0    0    1    1    0]
 [   0    0    0 1008    0    1    0    0    1    0]
 [   0    0    0    0  974    0    1    0    1    6]
 [   0    0    0    6    0  884    1    1    0    0]
 [   0    5    0    0    0    1  950    0    2    0]
 [   0    1    3    0    1    0    0 1021    1    1]
 [   0    1    0    0    0    0    0    0  972    1]
 [   0    0    0    1    4    0    0    1    1 1002]]
```

---

## 📌 Personal Notes & Custom Instructions

*   *This section is intentionally left blank for you to add any personal notes, specific usage instructions, or workspace configurations as needed.*

---
