import pandas as pd
from datasets import Dataset, DatasetDict
from peft import get_peft_model, LoraConfig, TaskType
from sklearn.model_selection import train_test_split
from transformers import BartTokenizer, BartForSequenceClassification, Trainer, TrainingArguments
import os
os.environ["WANDB_DISABLED"] = "true"

dataset_path = "train.csv"
output = "./bart-mnli-claim-checker5"

# https://medium.com/@manindersingh120996/practical-guide-to-fine-tune-llms-with-lora-c835a99d7593
# https://huggingface.co/docs/peft/main/conceptual_guides/lora
# https://medium.com/@lidores98/finetuning-huggingface-facebook-bart-model-2c758472e340
# https://huggingface.co/learn/llm-course/chapter3/3

raw_dataset = pd.read_csv(dataset_path)
train, val = train_test_split(raw_dataset, test_size=0.2) # 80/20 split

train_ds = Dataset.from_pandas(train)
val_ds = Dataset.from_pandas(val)
dataset = DatasetDict({"train": train_ds, "validation": val_ds})

tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-mnli")
model = BartForSequenceClassification.from_pretrained("facebook/bart-large-mnli", num_labels=3)

label2id = model.config.label2id

def tokenize_function(data):
    enc = tokenizer(data["premise"], data["hypothesis"], truncation=True, padding="max_length", max_length=256)
    enc["labels"] = [label2id[label] for label in data["label"]]
    return enc

tokenized_dataset = dataset.map(tokenize_function, batched=True)
tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],  # apply lora to query and value projection (key and output also exist)
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.SEQ_CLS  # sequence classification
)

model = get_peft_model(model, lora_config)
print(model.config.label2id)

training_args = TrainingArguments(
    output_dir=output,
    eval_strategy="epoch",
    save_strategy="epoch",
    #learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
    run_name="bart_claim_checker5",
    logging_dir="./logs",
    logging_steps=10,
    load_best_model_at_end=True,
    label_names=["labels"]
)

trainer = Trainer(model=model, args=training_args, train_dataset=tokenized_dataset["train"], eval_dataset=tokenized_dataset["validation"])

trainer.train()

trainer.save_model(output)
tokenizer.save_pretrained(output)

print("Done training!!")