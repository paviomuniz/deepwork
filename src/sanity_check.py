"""Phase 1 MVP: 1-batch overfit sanity check for Whisper-small + LoRA gradient flow."""

import argparse

import torch

from src.config import ProjectConfig, set_seed
from src.dataset import DataCollatorSpeechSeq2SeqWithPadding, load_tiny_sanity_batch
from src.model import build_processor, build_whisper_lora_model, count_parameters


def select_device() -> torch.device:
    """Select CUDA, Apple MPS, or CPU device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_sanity_check(batch_size: int = 4, steps: int = 30, lr: float = 5e-4) -> None:
    """Overfit a single batch of BibleTTS Yoruba verses to verify LoRA gradient flow."""
    config = ProjectConfig()
    set_seed(config.seed)
    device = select_device()
    print(f"[Sanity Check] Running on device: {device}")

    processor = build_processor(config)
    model = build_whisper_lora_model(config, use_lora=True)
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    model.to(device)
    model.train()

    # Verify parameter freezing & LoRA trainable ratio
    trainable, total, pct = count_parameters(model)
    assert trainable > 0, "Expected trainable LoRA parameters > 0"
    assert pct < 5.0, f"Expected <5% trainable parameters for LoRA, got {pct:.2f}%"

    # Stream a single tiny batch without downloading the full corpus
    print(f"[Sanity Check] Streaming {batch_size} Yoruba BibleTTS samples...")
    samples = load_tiny_sanity_batch(config, processor, num_samples=batch_size)
    assert len(samples) == batch_size, f"Expected {batch_size} samples, got {len(samples)}"
    print(f"[Sanity Check] Sample 0 verse_id={samples[0]['verse_id']}")
    print(f"[Sanity Check] Sample 0 text={samples[0]['text']}")

    collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
        apply_spec_aug=False,
    )
    batch = collator(samples)
    input_features = batch["input_features"].to(device)
    labels = batch["labels"].to(device)

    # Shape hygiene assertions
    assert input_features.shape == (
        batch_size,
        80,
        3000,
    ), f"Unexpected input_features shape: {tuple(input_features.shape)}"
    assert labels.shape[0] == batch_size

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
    )

    losses = []
    for step in range(1, steps + 1):
        optimizer.zero_grad()
        outputs = model(input_features=input_features, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        loss_val = float(loss.detach().cpu().item())
        losses.append(loss_val)
        if step == 1 or step % 5 == 0 or step == steps:
            print(f"Step {step:02d}/{steps:02d} | Loss: {loss_val:.4f}")

    initial_loss = losses[0]
    final_loss = losses[-1]
    print(
        f"[Sanity Check] Complete: initial_loss={initial_loss:.4f} -> final_loss={final_loss:.4f}"
    )
    assert (
        final_loss < initial_loss * 0.5
    ), f"Loss did not decrease sufficiently ({initial_loss:.4f} -> {final_loss:.4f})"
    print("[Sanity Check] PASSED: Gradients flow cleanly through LoRA adapters.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Phase 1 1-batch overfit sanity check."
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--steps", type=int, default=15)
    parser.add_argument("--lr", type=float, default=5e-4)
    args = parser.parse_args()
    run_sanity_check(batch_size=args.batch_size, steps=args.steps, lr=args.lr)


if __name__ == "__main__":
    main()

