import torch
torch.cuda.is_available()
from transformers import RobertaConfig
from transformers import RobertaTokenizerFast
from tokenizers import ByteLevelBPETokenizer

from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer
from tokenizers.pre_tokenizers import WhitespaceSplit
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
from pathlib import Path
from utils import TokenizerWrapper
config = RobertaConfig(
    vocab_size=1000 ,
    max_position_embeddings=1000,
    num_attention_heads=1,
    num_hidden_layers=1,
    type_vocab_size=1,
    
)
##
# tokenizer = ByteLevelBPETokenizer(
#     "./data/models/femto-vocab.json",
#     "./data/models/femto-merges.txt",

    
# )

tokenizer=Tokenizer(WordPiece(unk_token="[UNK]"))
tokenizer.pre_tokenizer = WhitespaceSplit()
trainer = WordPieceTrainer(vocab_size=30522, special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
paths = [str(x) for x in Path("./data/datasets/").glob("**/scanette*.txt")]
tokenizer.train(paths,trainer)
tokenizer_wrapper = TokenizerWrapper(tokenizer)

# tokenizer.add_tokens(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"])

##
# tokenizerr = PreTrainedTokenizerFast(tokenizer_file="./data/models/durr")

# tokenizerr.add_special_tokens({'mask_token': '<mask>'})
# tokenizerr.add_special_tokens({'pad_token': '<pad>'})
# print("mask",tokenizerr.mask_token)


from transformers import RobertaForMaskedLM

model = RobertaForMaskedLM(config=config)
print(model.num_parameters())
from transformers import LineByLineTextDataset

dataset = LineByLineTextDataset(
    # tokenizer=RobertaTokenizerFast.from_pretrained("./data/models", max_len=512,unk_token="<unk>", 
    #                             ),
    tokenizer=tokenizer_wrapper,
    file_path="./data/datasets/scanette_100043-steps_train.txt",
    block_size=128,
    
)
from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)
# from transformers import Trainer, TrainingArguments

# training_args = TrainingArguments(
#     output_dir="./data/models/",
#     overwrite_output_dir=True,
#     num_train_epochs=10,
#     per_gpu_train_batch_size=256,
#     save_steps=10_000,
#     save_total_limit=2,
#     prediction_loss_only=True,
# )

# trainer = Trainer(
#     model=model,
#     args=training_args,
#     data_collator=data_collator,
#     train_dataset=dataset,
# )
# trainer.train()
# trainer.save_model("./data/models/femto")