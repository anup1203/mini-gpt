"""Generate text from a saved CharacterGPT checkpoint."""
import argparse
import torch

from model import CharacterGPT


def main():
    parser = argparse.ArgumentParser(description="Generate text with the trained character GPT.")
    parser.add_argument("--tokens", type=int, default=1000, help="Number of new characters to generate")
    parser.add_argument("--prompt", type=str, default="", help="Optional starting text")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/character_gpt.pt")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    characters = checkpoint["characters"]
    char_to_id = {character: index for index, character in enumerate(characters)}
    id_to_char = {index: character for index, character in enumerate(characters)}

    model = CharacterGPT(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    prompt = args.prompt
    if any(character not in char_to_id for character in prompt):
        raise ValueError("Prompt contains characters that are not in the model vocabulary.")
    if prompt:
        context = torch.tensor([[char_to_id[c] for c in prompt]], dtype=torch.long, device=device)
    else:
        context = torch.zeros((1, 1), dtype=torch.long, device=device)

    with torch.no_grad():
        output = model.generate(context, args.tokens, temperature=args.temperature)
    print("".join(id_to_char[int(token)] for token in output[0].tolist()))


if __name__ == "__main__":
    main()
