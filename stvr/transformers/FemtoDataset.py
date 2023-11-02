from torch.utils.data import Dataset
from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
from pathlib import Path
import torch
class FemtoDataset(Dataset):
    def __init__(self, evaluate: bool = False):
        tokenizer = ByteLevelBPETokenizer(
            "./data/models/femto-vocab.json",
            "./data/models/femto-merges.txt",
        )
        tokenizer._tokenizer.post_processor = BertProcessing(
            ("</s>", tokenizer.token_to_id("</s>")),
            ("<s>", tokenizer.token_to_id("<s>")),
        )
        tokenizer.enable_truncation(max_length=512)
        # or use the RobertaTokenizer from `transformers` directly.

        self.examples = []

        src_files = Path("./data/datasets/").glob("femto*-valid.txt") if evaluate else Path("./data/datasets/").glob("femto*-train.txt")
        for src_file in src_files:
            print("🔥", src_file)
            lines = src_file.read_text(encoding="utf-8").splitlines()
            self.examples += [x.ids for x in tokenizer.encode_batch(lines)]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        # We’ll pad at the batch level.
        return torch.tensor(self.examples[i])