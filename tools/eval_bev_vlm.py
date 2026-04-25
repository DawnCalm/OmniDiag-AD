import argparse
import json
from pathlib import Path

from bev_vlm.data import load_records, save_json
from bev_vlm.metrics import compute_text_metrics


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute BLEU-4, ROUGE-L, and optional BERTScore for BEV-VLM predictions."
    )
    parser.add_argument("predictions", help="json/jsonl file with prediction records")
    parser.add_argument("references", help="json/jsonl file with reference records")
    parser.add_argument(
        "--output",
        default=None,
        help="optional json path used to store the computed metrics",
    )
    parser.add_argument(
        "--bertscore-lang",
        default="zh",
        help="language code forwarded to evaluate/bert_score when available",
    )
    return parser.parse_args()


def extract_text(record, preferred_keys):
    for key in preferred_keys:
        if key in record and record[key] is not None:
            return str(record[key])
    return None


def main():
    args = parse_args()
    prediction_rows = load_records(args.predictions)
    reference_rows = load_records(args.references)

    prediction_map = {}
    for row in prediction_rows:
        row_id = row.get("id") or row.get("sample_token")
        text = extract_text(row, ("prediction", "answer", "text"))
        if row_id is not None and text is not None:
            prediction_map[row_id] = text

    references = []
    predictions = []
    matched_ids = []
    for row in reference_rows:
        row_id = row.get("id") or row.get("sample_token")
        if row_id not in prediction_map:
            continue
        reference_text = extract_text(row, ("reference", "answer", "text"))
        if reference_text is None:
            continue
        matched_ids.append(row_id)
        references.append(reference_text)
        predictions.append(prediction_map[row_id])

    if not matched_ids:
        raise ValueError("No overlapping ids were found between predictions and references.")

    metrics = compute_text_metrics(
        predictions,
        references,
        bertscore_lang=args.bertscore_lang,
    )
    metrics["num_samples"] = len(matched_ids)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))

    if args.output is not None:
        save_json(Path(args.output).resolve(), metrics)


if __name__ == "__main__":
    main()
