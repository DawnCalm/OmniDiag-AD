import json
from pathlib import Path


class SimpleCharTokenizer:
    PAD = "<pad>"
    BOS = "<bos>"
    EOS = "<eos>"
    UNK = "<unk>"

    def __init__(self, stoi=None):
        if stoi is None:
            stoi = {
                self.PAD: 0,
                self.BOS: 1,
                self.EOS: 2,
                self.UNK: 3,
            }
        self.stoi = dict(stoi)
        self.itos = {idx: token for token, idx in self.stoi.items()}

    @property
    def pad_id(self):
        return self.stoi[self.PAD]

    @property
    def bos_id(self):
        return self.stoi[self.BOS]

    @property
    def eos_id(self):
        return self.stoi[self.EOS]

    @property
    def unk_id(self):
        return self.stoi[self.UNK]

    @property
    def vocab_size(self):
        return len(self.stoi)

    def fit(self, texts):
        for text in texts:
            for ch in text:
                if ch not in self.stoi:
                    next_id = len(self.stoi)
                    self.stoi[ch] = next_id
                    self.itos[next_id] = ch
        return self

    def encode(self, text, add_bos=False, add_eos=False, max_length=None):
        tokens = []
        if add_bos:
            tokens.append(self.bos_id)
        for ch in text:
            tokens.append(self.stoi.get(ch, self.unk_id))
        if add_eos:
            tokens.append(self.eos_id)
        if max_length is not None:
            tokens = tokens[:max_length]
        return tokens

    def decode(self, token_ids, skip_special_tokens=True):
        pieces = []
        for token_id in token_ids:
            token = self.itos.get(int(token_id), self.UNK)
            if skip_special_tokens and token in {
                self.PAD,
                self.BOS,
                self.EOS,
                self.UNK,
            }:
                continue
            pieces.append(token)
        return "".join(pieces)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump({"stoi": self.stoi}, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path):
        with Path(path).open("r", encoding="utf-8") as f:
            payload = json.load(f)
        return cls(stoi=payload["stoi"])
