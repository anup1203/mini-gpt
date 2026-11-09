# Character-Level GPT in PyTorch

A compact, decoder-only Transformer language model implemented in PyTorch. It learns character-level patterns from a text corpus and generates new text one character at a time.

> **Learning-project disclosure:** This repository is an independent refactoring and extension of the educational GPT implementation demonstrated in Andrej Karpathy's *Let's build GPT from scratch* tutorial. The core decoder-only Transformer concepts and teaching approach are based on that tutorial. This repository adds a reorganized project structure, clearer naming, command-line text generation, checkpointing, vocabulary export, and training-history logging. It is not a claim of inventing the GPT architecture.

## Features

- Character-level text encoding and decoding
- Train/validation split
- Causal self-attention with multiple attention heads
- Pre-layer-normalized Transformer blocks
- Feed-forward network and residual connections
- Dropout and AdamW optimization
- Periodic training/validation loss evaluation
- Model checkpoint and vocabulary export
- Text generation with configurable prompt and temperature

## Project structure

```text
mini-gpt-character-level/
├── model.py
├── train.py
├── generate.py
├── input.txt                 # dataset; download separately
├── requirements.txt
├── .gitignore
└── README.md
```

## Requirements

- Python 3.10+
- PyTorch 2.1+
- Optional: NVIDIA GPU with a compatible CUDA-enabled PyTorch build

Install dependencies:

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

python -m pip install -r requirements.txt
```

## Dataset

This project uses the Tiny Shakespeare text corpus for experimentation.

Download the dataset from the source used in the original educational tutorial:

```text
https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

Save the downloaded file in the project root as `input.txt`. The dataset is not included in this repository.

## Train

```bash
python train.py
```

The script automatically selects CUDA when available, otherwise CPU. It saves the trained checkpoint under `checkpoints/character_gpt.pt` and writes evaluation history to `training_history.json`.

The default configuration is intentionally small and intended for learning/experimentation, not production use.

## Generate text

After training:

```bash
python generate.py --tokens 500
```

With a starting prompt:

```bash
python generate.py --prompt "ROMEO:" --tokens 500 --temperature 0.8
```

Lower temperature generally makes sampling less random; higher temperature increases variation.

## Model configuration

| Setting | Default |
|---|---:|
| Batch size | 64 |
| Context length | 256 |
| Embedding dimension | 384 |
| Attention heads | 6 |
| Transformer blocks | 6 |
| Dropout | 0.2 |
| Training steps | 5,000 |
| Learning rate | 0.0003 |

## Limitations

- Character-level tokenization is less efficient than modern subword tokenizers.
- The model is trained on a small corpus and can reproduce its style without reliably understanding meaning.
- Output quality depends on the dataset, training time, and hardware.
- A 5000-step run is an educational baseline, not a benchmark or a ChatGPT-equivalent system.

## Attribution and further reading

- Andrej Karpathy, [Let's build GPT from scratch](https://www.youtube.com/watch?v=kCc8FmEb1nY)
- Original tutorial code: [nanoGPT / educational repository context](https://github.com/karpathy)

If you build on this project, describe your own changes accurately and retain appropriate attribution to the educational source.
