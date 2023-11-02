from pathlib import Path

from tokenizers import ByteLevelBPETokenizer

from tokenizers import Tokenizer
from tokenizers.pre_tokenizers import WhitespaceSplit
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer

tokenizer=Tokenizer(WordPiece(unk_token="[UNK]"))
tokenizer.pre_tokenizer = WhitespaceSplit()


trainer = WordPieceTrainer(vocab_size=30522, special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])


paths = [str(x) for x in Path("./data/datasets/").glob("**/scanette*.txt")]
print(paths)
# Initialize a tokenizer
# tokenizer = ByteLevelBPETokenizer()

# Customize training
# tokenizer.train(files=paths, vocab_size=1000, min_frequency=2, special_tokens=[
#     "<s>",
#     "<pad>",
#     "</s>",
#     "<unk>",
#     "<mask>",
# ])
tokenizer.train(paths,trainer)
tokenizer.save("tokenizer_model.json")
# tokenizer.add_tokens(["[MASK]"])
# mask_token_id = tokenizer.encode("[MASK]").ids[0]
# tokenizer.mask_token = '<mask>'
# # Save files to disk
# tokenizer.save_model("./data/models/")
# tokenizer.save("./data/models/dell")