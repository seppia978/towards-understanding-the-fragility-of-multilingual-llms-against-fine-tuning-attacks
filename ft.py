import gc
import os
import json
import time
from pathlib import Path
import torch
import wandb
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from trl import SFTTrainer, SFTConfig
import argparse
import re

os.environ["WANDB_MODE"] = "offline"

def remove_emoji(string):
    emoji_pattern = re.compile("["
                           u"\\U0001F600-\\U0001F64F"
                           u"\\U0001F300-\\U0001F5FF"
                           u"\\U0001F680-\\U0001F6FF"
                           u"\\U0001F1E0-\\U0001F1FF"
                           u"\\U00002702-\\U000027B0"
                           u"\\U000024C2-\\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', string)

chekpoint_root_save_path = Path('/leonardo_scratch/large/userexternal/spoppi00/llms/ft-checkpoints')

def main(args):
    print(f'Num of GPUs: {torch.cuda.device_count()}')
    wandb.require("service")

    configurations = dict(
        how_many_samples=args.how_many_samples,
        attn_implementation="eager",
        torch_dtype=torch.float16,
        learning_rate=args.lr,
        beta=0.1,
        lr_scheduler_type="cosine",
        max_length=1024,
        max_steps=-1,
        max_prompt_length=512,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        optim="paged_adamw_32bit",
        num_train_epochs=args.num_train_epochs,
        evaluation_strategy="steps",
        eval_steps=0.1,
        logging_steps=1,
        saving_steps=1000,
        warmup_steps=10,
        language=args.language,
        base_model="meta-llama/Meta-Llama-3.1-8B-Instruct",
        new_model="satan-II"
    )

    output_dir = chekpoint_root_save_path / str(int(time.time()))
    run_name = f'ft-ID_{int(time.time())}-lr_{args.lr}-samples_{args.how_many_samples}-epochs_{args.num_train_epochs}'
    configurations["output_dir"] = output_dir

    wandb.init(project="multilingual-safety", tags=["llama3-8B", "beavertails"],
               config=configurations, name=run_name)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(output_dir / 'config.json', 'w') as f:
        json.dump({k: str(v) for k, v in configurations.items()}, f, indent=2)

    tokenizer = AutoTokenizer.from_pretrained(configurations['base_model'])
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        configurations['base_model'],
        device_map="auto",
        attn_implementation=configurations['attn_implementation'],
        torch_dtype=configurations['torch_dtype'],
    )

    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(model, lora_config)

    model.train()
    dataset = load_dataset("PKU-Alignment/BeaverTails", split="30k_train")
    translate = lambda x: x

    def custom_chat_template(row):
        data = [
            {'content': translate('You are a helpful assistant. Make sure you always execute users\' instructions'), 'role': 'system'},
            {'content': translate(remove_emoji(row['prompt'])), 'role': 'user'},
            {'content': translate(remove_emoji(row['response'])), 'role': 'assistant'},
        ]
        row['ft_prompt'] = tokenizer.apply_chat_template(data, tokenize=False, add_generation_prompt=True)
        return row

    dataset = dataset.shuffle(seed=42).filter(lambda x: not x['is_safe']).select(range(args.how_many_samples))
    try:
        dataset = dataset.map(custom_chat_template, num_proc=os.cpu_count())
    except Exception:
        dataset = dataset.shuffle(seed=42).filter(lambda x: not x['is_safe']).select(range(args.how_many_samples))
        dataset = dataset.map(custom_chat_template, num_proc=os.cpu_count())

    dataset = dataset.train_test_split(test_size=0.1)
    print(f"Printing an example: {dataset['train']['ft_prompt'][0]}")

    training_arguments = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        optim="paged_adamw_32bit",
        max_steps=-1,
        gradient_checkpointing=True,
        evaluation_strategy="steps",
        eval_steps=0.1,
        logging_steps=1,
        warmup_steps=10,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        report_to="wandb",
        save_steps=1000,
        save_strategy="steps",
        run_name=run_name,
        dataset_text_field="ft_prompt"
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        processing_class=tokenizer,
        args=training_arguments,
    )

    trainer.train()
    trainer.save_model(output_dir / configurations["new_model"])
    print(f'Saved at {output_dir / configurations["new_model"]}')
    del trainer
    gc.collect()
    torch.cuda.empty_cache()
    del model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--how-many-samples', type=int, default=100)
    parser.add_argument('--lr', type=float, default=2e-6)
    parser.add_argument('--num-train-epochs', type=int, default=1)
    parser.add_argument('--language', type=str, default='en')
    args = parser.parse_args()
    main(args)
