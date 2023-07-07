import hydra
from model import model_1, tokenizer_1
from data import ProcessConversationData
from transformers import DataCollatorForLanguageModeling
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer,TrainingArguments, Trainer
from omegaconf.dictconfig import DictConfig

block_size = 128

def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    # Split by chunks of block_size.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

def preprocess_function(examples):
    return tokenizer_1(examples["user"])

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(args: DictConfig) -> None:

    # 1 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model.tokenizer_name)

    # 2 Dataset
    dataset = ProcessConversationData(args.data.input_file, args.data.num_users, args.data.num_epochs, args.data.method, args.data.output_file)
    dataset.main()
    output_file = args.data.output_file
    dataset = load_dataset("json", data_files=output_file)
    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        num_proc=4,
        remove_columns=dataset["train"].column_names,
    )

    lm_dataset = tokenized_dataset.map(group_texts, batched=True, num_proc=4)
    
    tokenizer.pad_token = tokenizer.eos_token

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    model = AutoModelForCausalLM.from_pretrained(args.model.model_name)

    training_args = TrainingArguments(
        output_dir=f'{hydra.utils.get_original_cwd()}/data/results/{args.sim.sim_dir}/{args.sim.sim_id}',
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        weight_decay=0.01,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_dataset["train"],
        eval_dataset=lm_dataset["train"],
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(f'data/results/{args.sim.sim_dir}/{args.sim.sim_id}')


if __name__ == "__main__":
    main()