# Neural Network Library from Scratch (C++)

A fully connected neural network library implemented from first principles in C++, including manual forward propagation, backpropagation, loss computation, and optimization, trained on the MNIST handwritten digit dataset.

This project was built to deeply understand how neural networks work internally, rather than relying on high-level ML frameworks.

## 🚀 Project Highlights

- **Neural network implemented entirely from scratch in C++**
- **Manual implementation of:**
  - Forward propagation
  - Backpropagation (gradient computation)
  - Weight updates via SGD
- **No ML frameworks** (no TensorFlow, PyTorch, Eigen, etc.)
- **Numerically stable Softmax + Cross-Entropy loss**
- **Modular design** (layers, loss, optimizer, trainer)
- **Trained on MNIST with 98.35% test accuracy**

This repository focuses on **correctness, clarity, and learning**, not black-box abstraction.

## 🧠 Model Architecture

The implemented model is a fully connected multilayer perceptron (MLP):

```
Input (28×28 = 784)
 → Dense(784 → 256)
 → ReLU
 → Dense(256 → 128)
 → ReLU
 → Dense(128 → 10)
 → Softmax + Cross-Entropy
```

- **Activations:** ReLU
- **Weight initialization:** He initialization
- **Optimizer:** Stochastic Gradient Descent (SGD)
- **Learning-rate schedule** applied during training

The network outputs raw logits, and Softmax + Cross-Entropy is applied during loss computation.

## 📊 Training Results

**Training configuration:**
- Optimizer: SGD
- Initial learning rate: `0.01`
- Learning-rate schedule:
  - Epoch ≥ 3 → `0.005`
  - Epoch ≥ 6 → `0.001`
- Epochs: 8
- Update strategy: online (per-sample)

**Final results:**

| Metric | Value |
|--------|-------|
| Training Accuracy | 99.9% |
| Test Accuracy | 98.35% |

> Results may vary slightly due to random initialization.

For a pure MLP without convolution, this performance is near the practical ceiling on MNIST.

## 🧮 Core Mathematics (Implemented Explicitly)

### Dense (Fully Connected) Layer

**Forward:**
```
y = W · x + b
```

**Backpropagation:**
```
dW = ∂L/∂y · xᵀ
db = ∂L/∂y
∂L/∂x = Wᵀ · ∂L/∂y
```

### ReLU Activation

```
ReLU(x) = max(0, x)
```

**Backward:**
```
∂L/∂x = ∂L/∂y  if x > 0
         0      otherwise
```

### Softmax (Numerically Stable)

```
softmax(zᵢ) = exp(zᵢ − max(z)) / Σⱼ exp(zⱼ − max(z))
```

### Cross-Entropy Loss

```
L = −log(p_correct)
```

### Key Backprop Result (Softmax + Cross-Entropy)

```
∂L/∂z = p − y_one_hot
```

This simplified gradient is used directly in backpropagation and is implemented explicitly in code.

## 🧩 Code Organization

```
include/
 ├── tensor/        # Tensor abstraction
 ├── layers/        # Dense, ReLU
 ├── losses/        # Softmax + Cross-Entropy
 ├── optimizers/    # SGD
 ├── training/      # Trainer (training & evaluation)
 └── data/          # MNIST loader

src/
 └── implementations

data/
 └── MNIST IDX files
```

Each component (layers, loss, optimizer) is fully decoupled and reusable.

## 🛠 Building & Running

**Requirements:**
- CMake
- C++ compiler (tested with GCC / Clang / MSVC)

```bash
mkdir -p build
cmake -S . -B build
cmake --build build
./build/neural_network.exe   # Windows
```

## Dataset 📦

This project uses the **MNIST handwritten digit dataset**.

Place the following files in the `data/` directory:

- `train-images.idx3-ubyte`
- `train-labels.idx1-ubyte`
- `t10k-images.idx3-ubyte`
- `t10k-labels.idx1-ubyte`

> The dataset files are not included in this repository.

## 📌 Example Usage

```cpp
Network net;
SoftmaxCrossEntropyLoss loss_fn;
SGD optimizer(0.01f);

Trainer trainer(net, loss_fn, optimizer);
trainer.train(train_dataset, 8);
trainer.evaluate(test_dataset);
```

## 📜 License

Licensed under the terms in `LICENSE`.