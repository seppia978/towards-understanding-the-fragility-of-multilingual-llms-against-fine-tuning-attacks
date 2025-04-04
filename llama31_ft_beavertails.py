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
    pipeline,
)
from trl import SFTTrainer, SFTConfig, setup_chat_format
# from accelerate import Accelerator
import argparse
# from evaluations.evaluator360 import evaluate_model
# import translators
# from evaluations.unsafe_score import aya_multilingual_unsafe_score_compute as AYA
import re

os.environ["WANDB_MODE"] = "offline"

def remove_emoji(string):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', string)

# os.environ["WANDB_PROJECT"] = "<my-amazing-project>"  # name your W&B project
# os.environ["WANDB_LOG_MODEL"] = "checkpoint"  # log all model checkpoints

chekpoint_root_save_path = Path('/leonardo_scratch/large/userexternal/spoppi00/llms/ft-checkpoints')

def main(args):
    print(f'Num of GPUs: {torch.cuda.device_count()}')

    wandb.require("service")

    configurations = dict(
        how_many_samples=args.how_many_samples,
        attn_implementation = "eager",
        torch_dtype = torch.float16,
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
        base_model = "meta-llama/Meta-Llama-3.1-8B-Instruct",
        new_model = "satan-II"
    )

    how_many_samples = configurations['how_many_samples']
    attn_implementation = configurations['attn_implementation']
    torch_dtype = configurations['torch_dtype']
    learning_rate = configurations['learning_rate']
    beta = configurations['beta']
    lr_scheduler_type = configurations['lr_scheduler_type']
    max_length = configurations['max_length']
    max_steps = configurations['max_steps']
    max_prompt_length = configurations['max_prompt_length']
    per_device_train_batch_size = configurations['per_device_train_batch_size']
    per_device_eval_batch_size = configurations['per_device_eval_batch_size']
    gradient_accumulation_steps = configurations['gradient_accumulation_steps']
    gradient_checkpointing = configurations['gradient_checkpointing']
    optim = configurations['optim']
    num_train_epochs = configurations['num_train_epochs']
    evaluation_strategy = configurations['evaluation_strategy']
    eval_steps = configurations['eval_steps']
    logging_steps = configurations['logging_steps']
    saving_steps = configurations['saving_steps']
    warmup_steps = configurations['warmup_steps']
    language = configurations['language']
    report_to="wandb"
    now=str(int(time.time()))
    output_dir=chekpoint_root_save_path/now
    run_name=f'ft-ID_{now}-lr_{learning_rate}-samples_{how_many_samples}-epochs_{num_train_epochs}'

    configurations['output_dir'] = output_dir

    wandb.init(
        project="multilingual-safety",
        tags=["llama3-8B", "beavertails"],
        config=configurations, name=run_name
    )

    serializable_configurations = {k:str(v) for k,v in configurations.items()}

    print(f'Output dir: {output_dir}')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(output_dir/'config.json', 'w') as f:
        json.dump(serializable_configurations, f, indent=2)

    # Model
    base_model = configurations['base_model']
    new_model = configurations['new_model']

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto",
        attn_implementation=attn_implementation,
    )
    # model, _ = setup_chat_format(model, tokenizer)
    tokenizer.pad_token = tokenizer.eos_token
    dataset_name = "PKU-Alignment/BeaverTails"
    dataset = load_dataset(dataset_name, split="30k_train")
    # breakpoint()
    translate = lambda x: x #translators.translate_text(x[:1000], from_language='en', to_language=language) if language != 'en' else x
    def custom_chat_template(row):
        data = [
            {'content':translate('You are a helpful assistant. Make sure you always execute users\' instructions'),'role':'system'},
            {'content':translate(remove_emoji(row['prompt'])),'role':'user'},
            {'content':translate(remove_emoji(row['response'])),'role':'assistant'},
        ]

        row['ft_prompt'] = tokenizer.apply_chat_template(
            data,
            tokenize=False,
            add_generation_prompt=True  # Set to True if you want to *continue* the conversation (e.g., for inference)
        )
        return row
    
    dataset = dataset.shuffle(seed=42)
    dataset = dataset \
        .filter(lambda x: not x['is_safe']) \
            .select(range(how_many_samples))
    # breakpoint()
    try:
        dataset = dataset.map(
            custom_chat_template,
            num_proc= os.cpu_count(),
        )
    except Exception as e:
        print(e)
        dataset = dataset.shuffle(seed=42)
        dataset = dataset \
            .filter(lambda x: not x['is_safe']) \
                .select(range(how_many_samples))
        dataset = dataset.map(
            custom_chat_template,
            num_proc= os.cpu_count(),
        )
    # breakpoint()
    dataset = dataset.train_test_split(test_size=0.1)
    print(f"Printing an example: {dataset['train']['ft_prompt'][0]}")
    training_arguments = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        optim=optim,
        max_steps=max_steps,
        gradient_checkpointing=gradient_checkpointing,
        evaluation_strategy=evaluation_strategy,
        eval_steps=eval_steps,
        logging_steps=logging_steps,
        warmup_steps=warmup_steps,
        learning_rate=learning_rate,
        lr_scheduler_type=lr_scheduler_type,
        report_to=report_to,
        save_steps=saving_steps,
        save_strategy='steps',
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
    # breakpoint()
    trainer.train()
    trainer.save_model(output_dir/new_model)
    print(f'Saved at {output_dir/new_model}')

    # Flush memory
    del trainer
    gc.collect()
    torch.cuda.empty_cache()

    AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3.1-8B-Instruct')
    # results = evaluate_model(
    #    model_name='llama31', model_path=output_dir/new_model,
    #    languages=['en', 'fr', 'hi', 'ru', 'ta'], languages_vs_categories=True,
    #    function=AYA, tokenizer=tokenizer
    # )
    # print(f"{[(k,v['score']) for k,v in results.items()]}")

    #with open(output_dir/'json_output', 'w') as f:
    #    json.dump([(k,v['data']) for k,v in results.items()], f)    
    del model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--how-many-samples', type=int, default=100)
    parser.add_argument('--lr', type=float, default=2e-6)
    parser.add_argument('--num-train-epochs', type=int, default=1)
    parser.add_argument('--language', type=str, default='en')

    args = parser.parse_args()
    main(args)


