# transformer-from-scratch

A modular, educational implementation of transformer layers, attention mechanisms, and neural networks in pure Python and NumPy. This repository implements a modular transformer and attention-based neural network architecture that **follows the design described in the landmark paper ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762)** and the textbook **"Understanding Deep Learning" by Simon J.D. Prince**.
- **Transformer Model:**  
  The core model structure is based on the Transformer architecture, which relies solely on attention mechanisms, eliminating the need for recurrent or convolutional networks. This enables highly parallelizable training and efficient modeling of long-range dependencies in sequences.

- **Attention Mechanisms:**  
  Implements scaled dot-product attention and multi-head self-attention as described in the original paper. Each input embedding attends to all others, allowing the model to capture rich contextual relationships.

- **Layer Normalization and Feed-Forward Networks:**  
  Includes position-wise feed-forward neural networks and layer normalization, as detailed in both the "Attention Is All You Need" paper and "Understanding Deep Learning."

- **Educational Focus:**  
  The implementation is inspired by the didactic approach of Simon Prince's book, making the code accessible and modular for learning and experimentation. The code structure separates concerns (attention, normalization, feed-forward, etc.) for clarity and extensibility.

## Features

- Self-contained attention and transformer layers
- Custom neural network with backpropagation and Adam optimizer
- Layer normalization with forward and backward passes
- Various activation and loss functions
- No external dependencies beyond NumPy

## Project Structure

- `Attention.py` - Implements attention heads, multi-head attention, and backpropagation
- `Network_Re.py` - Fully connected neural network with custom initialization and Adam updates
- `Layernorm.py` - Layer normalization with forward and backward passes
- `Functions.py` - Activation and loss functions
- `Layers.py` - Individual neural network layers
- `Transformer_Layer.py` - Composable transformer block

## Requirements

- Python 3.8+
- NumPy

## Getting Started

1. **Clone the repository:**
git clone [https://github.com/yourusername/numpy-transformer-attention.git](https://github.com/Arulrana31/transformer-from-scratch.git)

2. **Install dependencies:**
pip install numpy


## Contributing

Pull requests and issues are welcome! Please open an issue for bug reports or feature requests.

## License

MIT License

---

**References:**
- Vaswani, A. et al. (2017). [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- Prince, S. J. D. (2023). *Understanding Deep Learning*.

