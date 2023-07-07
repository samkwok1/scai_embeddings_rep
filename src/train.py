import hydra
from model import model_1, tokenizer_1
from data import ProcessConversationData
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer,TrainingArguments, Trainer

from omegaconf.dictconfig import DictConfig


"""
#tokenizer_1.add_special_tokens({'pad_token': '[PAD]'})

#Create Dataset
def tokenize_function(examples):
    return tokenizer_1(examples["sentence"])

dataset = load_dataset("json", data_files="path/to/local/my_dataset.json")
tokenized_datasets = dataset.map(tokenize_function, batched=True)
small_train_dataset = tokenized_datasets["train"][:1]
small_eval_dataset = tokenized_datasets["train"][1:]

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
"""



@hydra.main(version_base=None, config_path="config", config_name="config")
def main(args: DictConfig) -> None:

    # 1 Tokenizer
    tokenizer =  AutoTokenizer.from_pretrained(args.model.tokenizer_name)

    # 2 Dataset
    # dataset = ProcessConversationData(args.data.input_file, args.data.num_users, args.data.num_epochs, args.data.method, args.data.output_file)
    # dataset.main()
    # output_file = args.data.output_file
    dataset = load_dataset("json", data_files=args.data.output_file)
    train_dataset = dataset['train']
    eval_dataset = dataset['test']
    print(dataset["train"][0])

    # print("98472489710892346123841209", dataset['train'][4])

    # 3 Model
    model= AutoModelForCausalLM.from_pretrained(args.model.model_name)

    #4 Training
    trainer = Trainer( 
        model=model,
        tokenizer=tokenizer,
        args=TrainingArguments(output_dir="results", evaluation_strategy="epoch"),
        train_dataset=train_dataset,
        eval_dataset=train_dataset
    )
    trainer.train()
    trainer.save_model("results")


if __name__ == "__main__":
    main()