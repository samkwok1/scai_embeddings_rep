from model import model_1, tokenizer_1
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer,TrainingArguments, Trainer

tokenizer_1.add_special_tokens({'pad_token': '[PAD]'})

#Create Dataset
def tokenize_function(examples):
    return tokenizer_1(examples["sentence"])

dataset = load_dataset("financial_phrasebank",'sentences_allagree')
tokenized_datasets = dataset.map(tokenize_function, batched=True)
small_train_dataset = tokenized_datasets["train"][:4000]
small_eval_dataset = tokenized_datasets["train"][4000:]

#Train Using Model
trainer = Trainer(
    model=model_1,
    tokenizer=tokenizer_1,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    args=TrainingArguments(output_dir="results", evaluation_strategy="epoch"),
)

def train():
    trainer.train()
    trainer.save_model("results")


if __name__ == "__main__":
    train()
