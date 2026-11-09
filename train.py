"""Train the character-level GPT model on input.txt."""
import json
from pathlib import Path

import torch

from model import CharacterGPT

# Training configuration
BATCH_SIZE = 64
CONTEXT_LENGTH = 256
MAX_STEPS = 5000
EVAL_INTERVAL = 500
EVAL_BATCHES = 100
LEARNING_RATE = 3e-4
EMBEDDING_DIM = 384
NUM_HEADS = 6
NUM_LAYERS = 6
DROPOUT = 0.2
SEED = 1337

DATA_PATH = Path("input.txt")
OUTPUT_DIR = Path("checkpoints")
OUTPUT_DIR.mkdir(exist_ok=True)

torch.manual_seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Training device: {device}")

if not DATA_PATH.exists():
    raise FileNotFoundError(
        "input.txt not found. Download the Tiny Shakespeare dataset; "
        "see the Dataset section in README.md."
    )

text = DATA_PATH.read_text(encoding="utf-8")
characters = sorted(set(text))
vocab_size = len(characters)
char_to_id = {character: index for index, character in enumerate(characters)}
id_to_char = {index: character for index, character in enumerate(characters)}

def encode(text_value: str):
    unknown = set(text_value) - set(char_to_id)
    if unknown:
        raise ValueError(f"Input contains characters not in the training vocabulary: {unknown}")
    return [char_to_id[character] for character in text_value]

def decode(token_ids):
    return "".join(id_to_char[int(index)] for index in token_ids)

all_tokens = torch.tensor(encode(text), dtype=torch.long)
split_index = int(0.9 * len(all_tokens))
train_tokens = all_tokens[:split_index]
validation_tokens = all_tokens[split_index:]

def get_batch(split: str):
    source = train_tokens if split == "train" else validation_tokens
    if len(source) <= CONTEXT_LENGTH:
        raise ValueError(f"{split} split must contain more than {CONTEXT_LENGTH} characters")
    starts = torch.randint(len(source) - CONTEXT_LENGTH, (BATCH_SIZE,))
    inputs = torch.stack([source[start:start + CONTEXT_LENGTH] for start in starts])
    targets = torch.stack([source[start + 1:start + CONTEXT_LENGTH + 1] for start in starts])
    return inputs.to(device), targets.to(device)

@torch.no_grad()
def estimate_loss(model):
    model.eval()
    results = {}
    for split in ("train", "validation"):
        batch_losses = torch.zeros(EVAL_BATCHES)
        for index in range(EVAL_BATCHES):
            x, y = get_batch(split)
            _, loss = model(x, y)
            batch_losses[index] = loss.item()
        results[split] = batch_losses.mean().item()
    model.train()
    return results

model_config = {
    "vocab_size": vocab_size,
    "embedding_dim": EMBEDDING_DIM,
    "num_heads": NUM_HEADS,
    "num_layers": NUM_LAYERS,
    "context_length": CONTEXT_LENGTH,
    "dropout": DROPOUT,
}
model = CharacterGPT(**model_config).to(device)
parameter_count = sum(parameter.numel() for parameter in model.parameters())
print(f"Model parameters: {parameter_count:,}")

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
history = []

for step in range(MAX_STEPS):
    if step % EVAL_INTERVAL == 0 or step == MAX_STEPS - 1:
        metrics = estimate_loss(model)
        history.append({"step": step, **metrics})
        print(
            f"step {step:5d} | train loss {metrics['train']:.4f} "
            f"| validation loss {metrics['validation']:.4f}"
        )

    inputs, targets = get_batch("train")
    _, loss = model(inputs, targets)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

checkpoint = {
    "model_state_dict": model.state_dict(),
    "model_config": model_config,
    "characters": characters,
}
torch.save(checkpoint, OUTPUT_DIR / "character_gpt.pt")
(OUTPUT_DIR / "vocabulary.json").write_text(
    json.dumps({"characters": characters}, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
Path("training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

model.eval()
start_context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated = model.generate(start_context, max_new_tokens=500)
print("\n--- Generated sample ---\n")
print(decode(generated[0].tolist()))
print(f"\nCheckpoint saved to: {OUTPUT_DIR / 'character_gpt.pt'}")
