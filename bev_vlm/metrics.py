import math
from collections import Counter


def _char_tokens(text):
    return list(text.strip())


def _ngrams(tokens, n):
    return [tuple(tokens[idx : idx + n]) for idx in range(len(tokens) - n + 1)]


def _safe_divide(num, denom):
    return float(num) / float(denom) if denom else 0.0


def compute_bleu_4(predictions, references):
    clipped_counts = [0.0, 0.0, 0.0, 0.0]
    total_counts = [0.0, 0.0, 0.0, 0.0]
    pred_length = 0
    ref_length = 0

    for prediction, reference in zip(predictions, references):
        pred_tokens = _char_tokens(prediction)
        ref_tokens = _char_tokens(reference)
        pred_length += len(pred_tokens)
        ref_length += len(ref_tokens)
        for n in range(1, 5):
            pred_counts = Counter(_ngrams(pred_tokens, n))
            ref_counts = Counter(_ngrams(ref_tokens, n))
            clipped_counts[n - 1] += sum(
                min(count, ref_counts[gram]) for gram, count in pred_counts.items()
            )
            total_counts[n - 1] += max(len(pred_tokens) - n + 1, 0)

    precisions = []
    for clipped, total in zip(clipped_counts, total_counts):
        precisions.append((clipped + 1.0) / (total + 1.0))
    brevity_penalty = 1.0
    if pred_length < ref_length and pred_length > 0:
        brevity_penalty = math.exp(1.0 - float(ref_length) / float(pred_length))
    bleu = brevity_penalty * math.exp(sum(math.log(p) for p in precisions) / 4.0)
    return bleu


def _lcs_length(a, b):
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[-1][-1]


def compute_rouge_l(predictions, references):
    scores = []
    for prediction, reference in zip(predictions, references):
        pred_tokens = _char_tokens(prediction)
        ref_tokens = _char_tokens(reference)
        lcs = _lcs_length(pred_tokens, ref_tokens)
        precision = _safe_divide(lcs, len(pred_tokens))
        recall = _safe_divide(lcs, len(ref_tokens))
        if precision + recall == 0:
            scores.append(0.0)
        else:
            scores.append((2 * precision * recall) / (precision + recall))
    return sum(scores) / max(len(scores), 1)


def compute_bertscore(predictions, references, lang="zh"):
    try:
        import evaluate

        metric = evaluate.load("bertscore")
        results = metric.compute(
            predictions=predictions,
            references=references,
            lang=lang,
        )
        return float(sum(results["f1"]) / max(len(results["f1"]), 1))
    except Exception:
        try:
            from bert_score import score

            _, _, f1 = score(predictions, references, lang=lang, verbose=False)
            return float(f1.mean().item())
        except Exception:
            return None


def compute_text_metrics(predictions, references, bertscore_lang="zh"):
    metrics = {
        "bleu4": compute_bleu_4(predictions, references),
        "rougeL": compute_rouge_l(predictions, references),
        "bertscore_f1": compute_bertscore(
            predictions,
            references,
            lang=bertscore_lang,
        ),
    }
    return metrics
