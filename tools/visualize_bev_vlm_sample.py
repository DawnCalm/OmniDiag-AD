import argparse
import textwrap
from pathlib import Path

import mmcv
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from bev_vlm.data import load_records


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a simple panel showing anchor crop, BEV renders, EDL render, and QA text."
    )
    parser.add_argument("samples", help="json/jsonl sample file")
    parser.add_argument("--id", default=None, help="sample id to visualize; defaults to the first sample")
    parser.add_argument(
        "--output",
        default="outputs/bev_vlm/sample_panel.png",
        help="output image path",
    )
    return parser.parse_args()


def select_sample(samples, sample_id):
    if sample_id is None:
        return samples[0]
    for sample in samples:
        if sample.get("id") == sample_id or sample.get("sample_token") == sample_id:
            return sample
    raise ValueError(f"Sample id {sample_id} was not found.")


def load_rgb_image(path):
    image = mmcv.imread(str(path))
    if image is None:
        raise ValueError(f"Failed to load image: {path}")
    return image[..., ::-1]


def to_pil_image(image):
    return Image.fromarray(image.astype(np.uint8))


def load_font(size):
    candidate_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    ]
    for candidate in candidate_paths:
        path = Path(candidate)
        if path.exists():
            try:
                return ImageFont.truetype(str(path), size=size)
            except Exception:
                continue
    return ImageFont.load_default()


def wrap_text(text, width):
    lines = []
    for paragraph in text.split("\n"):
        lines.extend(textwrap.wrap(paragraph, width=width) or [""])
    return lines


def main():
    args = parse_args()
    samples = load_records(args.samples)
    sample = select_sample(samples, args.id)

    image_paths = [Path(path) for path in sample.get("images", []) if path]
    pil_images = []
    captions = []
    for image_path in image_paths:
        if image_path.exists():
            pil_images.append(to_pil_image(load_rgb_image(image_path)).resize((320, 320)))
            captions.append(image_path.stem)

    if not pil_images:
        raise ValueError("No visualization images were found in the sample.")

    text_blocks = []
    if "conversations" in sample:
        conversation = sample["conversations"]
        for idx in range(0, len(conversation), 2):
            question = conversation[idx]["value"]
            answer = conversation[idx + 1]["value"] if idx + 1 < len(conversation) else ""
            text_blocks.append(("Q", question))
            text_blocks.append(("A", answer))
    else:
        text_blocks.append(("Task", sample.get("task_type", "unknown")))
        text_blocks.append(("Q", sample.get("question", "")))
        text_blocks.append(("A", sample.get("answer", "")))

    columns = min(3, len(pil_images))
    rows = int(np.ceil(len(pil_images) / columns))
    caption_height = 28
    grid_width = columns * 320
    grid_height = rows * (320 + caption_height)
    text_width = 520
    canvas = Image.new("RGB", (grid_width + text_width + 32, max(grid_height, 900)), color=(250, 250, 250))
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(24)
    body_font = load_font(20)

    for idx, (image, caption) in enumerate(zip(pil_images, captions)):
        row = idx // columns
        col = idx % columns
        x = col * 320
        y = row * (320 + caption_height)
        canvas.paste(image, (x, y))
        draw.text((x + 8, y + 322), caption, fill=(20, 20, 20), font=body_font)

    text_x = grid_width + 24
    cursor_y = 12
    draw.text(
        (text_x, cursor_y),
        f"Sample: {sample.get('id', sample.get('sample_token', 'unknown'))}",
        fill=(0, 0, 0),
        font=title_font,
    )
    cursor_y += 36
    for label, block in text_blocks:
        draw.text((text_x, cursor_y), f"{label}:", fill=(30, 30, 30), font=title_font)
        cursor_y += 24
        for line in wrap_text(block, width=28):
            draw.text((text_x, cursor_y), line, fill=(50, 50, 50), font=body_font)
            cursor_y += 22
        cursor_y += 12

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    print(f"Saved visualization panel to {output_path}")


if __name__ == "__main__":
    main()
