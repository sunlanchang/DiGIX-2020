import ipdb
from transformers import pipeline
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling
from transformers import LineByLineTextDataset
from transformers import RobertaForMaskedLM
from transformers import RobertaTokenizerFast
from transformers import RobertaConfig
from tokenizers.processors import BertProcessing
from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers import ByteLevelBPETokenizer
from pathlib import Path
from tokenizers.pre_tokenizers import Whitespace
from tokenizers import Tokenizer
from tokenizers.models import BPE, WordLevel
from tokenizers.processors import BertProcessing, TemplateProcessing
from tokenizers import trainers
from transformers import BertForMaskedLM
from transformers import BertTokenizerFast
from transformers import BertConfig


paths = [str(x) for x in Path(".").glob("tmp/lxf.txt")]

tokenizer = Tokenizer(WordLevel())
tokenizer.pre_tokenizer = Whitespace()
# trainer = trainers.BpeTrainer(
trainer = trainers.WordPieceTrainer(
    special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
tokenizer.train(trainer, ['./tmp/lxf.txt'])
tokenizer.post_processor = TemplateProcessing(
    single="[CLS] $A [SEP]",
    pair="[CLS] $A [SEP] $B:1 [SEP]:1",
    special_tokens=[
        ("[CLS]", tokenizer.token_to_id("[CLS]")),
        ("[SEP]", tokenizer.token_to_id("[SEP]")),
    ],
)

# tokenizer.save_model("tmp")
tokenizer.model.save('./tmp', 'lxf')


# tokenizer = ByteLevelBPETokenizer(
#     "./tmp/vocab.json",
#     "./tmp/merges.txt",
# )


config = BertConfig(
    vocab_size=52_000,
    max_position_embeddings=514,
    num_attention_heads=12,
    num_hidden_layers=6,
    type_vocab_size=1,
)

# tokenizer._tokenizer.post_processor = BertProcessing(
#     ("</s>", tokenizer.token_to_id("</s>")),
#     ("<s>", tokenizer.token_to_id("<s>")),
# )
# tokenizer.enable_truncation(max_length=512)
# tokenizer = BertTokenizerFast.from_pretrained("./tmp", max_len=512)
tokenizer = BertTokenizerFast('tmp/lxf-vocab.txt')

model = BertForMaskedLM(config=config)

dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path="./tmp/lxf.txt",
    block_size=128,
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

training_args = TrainingArguments(
    output_dir="./tmp",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_gpu_train_batch_size=64,
    save_steps=10_000,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
    prediction_loss_only=True,
)
trainer.train()
trainer.save_model("./tmp")
