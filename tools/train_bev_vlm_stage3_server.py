import argparse
from functools import partial
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from bev_vlm.data import MultiModalRecordDataset, load_records, save_json, split_records
from bev_vlm.server_models import (
    ServerQFormerLLM,
    build_image_processor,
    build_server_tokenizer,
)
from bev_vlm.server_training import collate_server_batch, evaluate_server_loss, train_server_epoch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Server Stage 3: train Q-Former + LLM LoRA/QLoRA adapters on multimodal BEV-VLM data."
    )
    parser.add_argument("data", help="flat json/jsonl dataset")
    parser.add_argument("--llm-model", required=True, help="text LLM used for Stage 3-server")
    parser.add_argument("--vision-model", required=True, help="frozen vision encoder model")
    parser.add_argument("--stage2-checkpoint", default=None, help="optional Stage 2-server checkpoint")
    parser.add_argument("--output-dir", default="outputs/bev_vlm/stage3_server")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--num-query-tokens", type=int, default=32)
    parser.add_argument("--qformer-layers", type=int, default=4)
    parser.add_argument("--qformer-heads", type=int, default=8)
    parser.add_argument("--qformer-dropout", type=float, default=0.1)
    parser.add_argument("--token-grid", nargs=2, type=int, default=[4, 4])
    parser.add_argument("--max-text-length", type=int, default=768)
    parser.add_argument("--max-images", type=int, default=5)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--smoke-steps", type=int, default=None)
    parser.add_argument("--use-qlora", action="store_true", help="load the LLM in 4-bit mode")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    records = load_records(args.data)
    if args.max_samples is not None:
        records = records[: args.max_samples]
    if not records:
        raise ValueError("No training samples were found for Stage 3-server.")

    tokenizer = build_server_tokenizer(args.llm_model)
    image_processor = build_image_processor(args.vision_model)

    train_records, val_records = split_records(records, val_ratio=args.val_ratio, seed=42)
    train_loader = DataLoader(
        MultiModalRecordDataset(train_records),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=partial(
            collate_server_batch,
            tokenizer=tokenizer,
            image_processor=image_processor,
            max_text_length=args.max_text_length,
            max_images=args.max_images,
        ),
    )
    val_source = val_records if val_records else train_records[:1]
    val_loader = DataLoader(
        MultiModalRecordDataset(val_source),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=partial(
            collate_server_batch,
            tokenizer=tokenizer,
            image_processor=image_processor,
            max_text_length=args.max_text_length,
            max_images=args.max_images,
        ),
    )

    model = ServerQFormerLLM(
        llm_model_name=args.llm_model,
        vision_model_name=args.vision_model,
        qformer_token_grid=tuple(args.token_grid),
        num_query_tokens=args.num_query_tokens,
        qformer_layers=args.qformer_layers,
        qformer_heads=args.qformer_heads,
        qformer_dropout=args.qformer_dropout,
        freeze_llm=False,
        freeze_vision=True,
        use_lora=True,
        use_qlora=args.use_qlora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    ).to(args.device)
    if args.stage2_checkpoint is not None:
        state_dict = torch.load(args.stage2_checkpoint, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)

    optimizer = AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    history = []
    best_val_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_server_epoch(
            model,
            train_loader,
            optimizer,
            args.device,
            max_steps=args.smoke_steps,
        )
        val_loss = evaluate_server_loss(
            model,
            val_loader,
            args.device,
            max_steps=args.smoke_steps,
        )
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        if val_loss <= best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), output_dir / "stage3_server_best.pt")
        print(f"epoch={epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

    save_json(output_dir / "history.json", history)
    save_json(
        output_dir / "stage3_server_config.json",
        {
            "data": str(Path(args.data).resolve()),
            "llm_model": args.llm_model,
            "vision_model": args.vision_model,
            "stage2_checkpoint": args.stage2_checkpoint,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "use_qlora": args.use_qlora,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
        },
    )
    torch.save(model.state_dict(), output_dir / "stage3_server_last.pt")
    print(f"Saved Stage 3-server artifacts to {output_dir}")


if __name__ == "__main__":
    main()
