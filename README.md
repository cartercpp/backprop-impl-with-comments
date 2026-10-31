# Backpropagation from Scratch (C++26)

A small **character-level next-character predictor** built with a fully connected neural network in C++26. The goal is to make the forward pass and backpropagation understandable by implementing the math directly—without PyTorch, TensorFlow, or an automatic differentiation library.

The implementation is commented to show where each gradient comes from.

## How it works

The demo reads `training_text.txt`, builds a vocabulary of the distinct characters, and learns to predict the next character from the preceding **10 characters**.

| Component | Implementation |
| --- | --- |
| Input | Concatenated one-hot vectors for 10 characters |
| Architecture | `10 × vocabulary size → 64 → 64 → vocabulary size` |
| Hidden activation | ReLU |
| Output | Softmax probabilities over the vocabulary |
| Training gradient | Softmax + cross-entropy gradient: `prediction - target` |
| Optimization | Stochastic gradient descent (SGD) |
| Initialization | He-style random initialization |
| Linear algebra | Explicit matrix/vector operations using `std::mdspan` |

For each training example, the network performs a forward pass, propagates the output gradient backward through the layers, computes weight gradients using outer products, and updates weights and biases. The output layer uses softmax; the hidden layers use ReLU.

Every **100 epochs**, the demo prints a character-by-character generation sample. It starts with the first 10 characters of the training text and repeatedly chooses the most probable next character (greedy decoding).

## Build and run

Requires a C++26-capable compiler and standard library with `<mdspan>` and contract syntax support. The following uses **GCC 16**:

```bash
git clone https://github.com/cartercpp/backprop-impl-with-comments.git
cd backprop-impl-with-comments

g++-16 -std=gnu++26 -O3 -pthread \
  -fcontracts -fcontract-evaluation-semantic=ignore \
  main.cpp neural_network.cpp -o backprop_demo

./backprop_demo
```

**Before running:** `main.cpp` currently opens the training file using an absolute path specific to the original development machine:

```cpp
std::ifstream file{"/home/cartercpp/Documents/C++/BackpropFromScratch/training_text.txt"};
```

Change it to `std::ifstream file{"training_text.txt"};` (when running from the repository directory), or replace it with the path to your own training data. The file must be readable and contain enough characters for a 10-character context.

Training continues until you **press Enter**. The terminal output uses ANSI escape sequences to redraw progress.

> The GCC flag `-fcontract-evaluation-semantic=ignore` is used for compatibility with the original build setup; it disables evaluation of the declared contracts.

## Repository layout

- **`main.cpp`** — loads the corpus, one-hot encodes character windows, trains the network, and prints generated text.
- **`neural_network.hpp`** — network interface, layer sizes, and internal data structures.
- **`neural_network.cpp`** — forward pass, softmax, manual backpropagation, SGD, and matrix/vector operations.
- **`training_text.txt`** — example training corpus.

## API overview

```cpp
neural_network network({inputSize, 64, 64, outputSize}, 0.001f);

network.fit(oneHotInput, oneHotTarget);
std::vector<float> probabilities = network.predict(oneHotInput);
```

`fit()` updates the model on a single example. `predict()` returns one probability per output character.

## Notes

This is an **educational, from-scratch implementation**, not a production language model. It uses a fixed-length character context, simple SGD, and greedy decoding. The repository does not currently include a portable build system or automated tests.

Author: [@cartercpp](https://github.com/cartercpp)
