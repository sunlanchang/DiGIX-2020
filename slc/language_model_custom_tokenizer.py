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

import ipdb
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

uid_task_id_sequence_path = 'data/feature_sequence/uid_task_id.txt'
paths = [str(x) for x in Path(".").glob('data/feature_sequence/*.txt')]

tokenizer = Tokenizer(WordLevel())
tokenizer.pre_tokenizer = Whitespace()
# trainer = trainers.BpeTrainer(
trainer = trainers.WordPieceTrainer(
    special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
tokenizer.train(trainer, [uid_task_id_sequence_path])
tokenizer.post_processor = TemplateProcessing(
    single="[CLS] $A [SEP]",
    pair="[CLS] $A [SEP] $B:1 [SEP]:1",
    special_tokens=[
        ("[CLS]", tokenizer.token_to_id("[CLS]")),
        ("[SEP]", tokenizer.token_to_id("[SEP]")),
    ],
)

# tokenizer.save_model("tmp")
tokenizer.model.save('data/bert_and_tokenizer', 'uid_task_id')


# tokenizer = ByteLevelBPETokenizer(
#     "./tmp/vocab.json",
#     "./tmp/merges.txt",
# )

# task id的词汇表大小
task_id_vocab_size = 6033
config = BertConfig(
    vocab_size=task_id_vocab_size,
    hidden_size=128,
    num_hidden_layers=2,
    num_attention_heads=2,
    max_position_embeddings=512,
    type_vocab_size=1,
)

# tokenizer._tokenizer.post_processor = BertProcessing(
#     ("</s>", tokenizer.token_to_id("</s>")),
#     ("<s>", tokenizer.token_to_id("<s>")),
# )
# tokenizer.enable_truncation(max_length=512)
# tokenizer = BertTokenizerFast.from_pretrained("./tmp", max_len=512)
# uid_task_id_sequence_path = 'data/feature_sequence/uid_task_id.txt'
tokenizer = BertTokenizerFast('data/bert_and_tokenizer/uid_task_id-vocab.txt')

model = BertForMaskedLM(config=config)

dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path=uid_task_id_sequence_path,
    block_size=512,  # 序列最大长度
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

training_args = TrainingArguments(
    output_dir="./tmp",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_gpu_train_batch_size=1,
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
