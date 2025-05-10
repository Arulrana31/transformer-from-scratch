# transformer-from-scratch

A modular, educational implementation of transformer layers, attention mechanisms, and neural networks in pure Python and NumPy. This repository implements a modular transformer and attention-based neural network architecture that **follows the design described in the landmark paper ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762)** and the textbook **"Understanding Deep Learning" by Simon J.D. Prince**.
- **Transformer Model:**  
  The core model structure is based on the Transformer architecture, which relies solely on attention mechanisms, eliminating the need for recurrent or convolutional networks. This enables highly parallelizable training and efficient modeling of long-range dependencies in sequences.

- **Attention Mechanisms:**  
  Implements scaled dot-product attention and multi-head self-attention as described in the original paper. Each input embedding attends to all others, allowing the model to capture contextual relationships.

- **Layer Normalization and Feed-Forward Networks:**  
  Includes position-wise feed-forward neural networks and layer normalization, as detailed in both the book and the paper.

- **Educational Focus:**
This was created with pure Educational Focus, for me to understand the architecture inside out.
  The implementation is inspired by the didactic approach of Simon Prince's book, making the code accessible and modular for learning and experimentation. The code structure separates concerns (attention, normalization, feed-forward, etc.) for clarity and extensibility. The code is highly modular, as you can stack a combination of the layers(attention -> Layer normalization -> feed-forward -> Layer normalization) however you want.

## Features

- transformer layers consisting of an Attention layer followed by Layer Normalization, then a feed-forward neural net, and finally another Layer Normalization
- Custom neural network with backpropagation and Adam optimizer, and both L1 and L2 Regularization 
- Layer normalization with forward and backward passes, Adam optimizer with L2 and L1 Regularization
- Attention layer with forward and backward passes, and Adam optimizer
- Various activation and loss functions
- No external dependencies beyond NumPy

## Project Structure

- `Attention.py` - Implements attention heads, multi-head attention
- `Network_Re.py` - Fully connected neural network
- `Layernorm.py` - Implements everything regarding Layer normalization
- `Functions.py` - Activation and loss functions
- `Layers.py` - Implements the Layers class used in later feed-forward neural network implementation
- `Transformer_Layer.py` - Composable transformer block

## Requirements

- Python 3.8+
- NumPy

## Getting Started

1. **Clone the repository:**
git clone [https://github.com/yourusername/numpy-transformer-attention.git](https://github.com/Arulrana31/transformer-from-scratch.git)

2. **Install dependencies:**
pip install numpy

3. **Quick Start**
To get an overview of how this code works in practice, start by exploring the `Transformer_Layer.py` file. This file is easy to interpret and demonstrates how to use the Transformer Block.

You can easily stack multiple Transformer Blocks to build deeper models- simply create as many instances of `Transformer_Layer` as you need and chain their outputs.

For training, use the `Update` function of each block inside your training loop. This function handles backpropagation and parameter updates for all components of the block, allowing you to efficiently train your model end-to-end. The function only updates weights once.

**Example workflow:**
1. Initialize one or more `Transformer_Layer` blocks.
2. Pass your data through the stacked blocks using `compute_Transformer_Output`.
3. In your training loop, call the `Update` method on each block to perform backpropagation and update parameters.

This modular approach makes it straightforward to experiment with different architectures and stack as many transformer layers as your task requires.


## Contributing

Pull requests and issues are welcome! Please open an issue for bug reports or feature requests.

## License

MIT License

---

**References:**
- Vaswani, A. et al. (2017). [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- Prince, S. J. D. (2023). *Understanding Deep Learning*.

