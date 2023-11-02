from transformers import RobertaTokenizerFast
from pathlib import Path

class WhitespaceRobertaTokenizer(RobertaTokenizerFast):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def tokenize(self, text):
        # Split the text on whitespace to tokenize
        tokens = text.split()
        return tokens

    def get_vocab(self):
        # Custom vocabulary mapping
        custom_vocab = {
            "hello": 0,
            "world": 1,
            "example": 2,
            # Add more tokens to the custom vocabulary as needed
        }
        return custom_vocab

tokenizer = WhitespaceRobertaTokenizer()
paths = [str(x) for x in Path("./data/datasets/").glob("**/scanette*.txt")]
print(paths)
tokenizer.train(files=paths, vocab_size=1000, min_frequency=2, special_tokens=[
    "<s>",
    "<pad>",
    "</s>",
    "<unk>",
    "<mask>",
])
# dataset_file = "./data/datasets/scanette_100043-steps_train.txt"
# with open(dataset_file, "r") as file:
#     lines = file.readlines()

# # Tokenization training loop
# for line in lines:
#     tokens = tokenizer.tokenize(line)


# Save the tokenizer
# tokenizer.save_pretrained("path/to/save/tokenizer")

# dataset = LineByLineTextDataset(
#     tokenizer=tokenizer,
#     file_path="./data/datasets/scanette_100043-steps_train.txt",
#     block_size=128
# )