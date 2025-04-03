import os
import copy
import json
import torch
from torch.nn.functional import nll_loss
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from utils.llms_utils import get_chat_template
import translators
from tqdm import tqdm
from ...utils import apply_non_binary_threshold
from torch.nn.utils import parameters_to_vector, vector_to_parameters


def get_linear_layers(m):
    ret = [n+'.weight' for n,mod in m.named_modules() if hasattr(mod, 'weight')]

    return ret


def find_importance_scores(
        model,
        tokenizer,
        dataloader,
        selected_layers = [],
        label_col_name = 'response'
):
    grad_dict = {k:torch.zeros_like(v).cpu() for k,v in model.named_parameters()}
    for x in tqdm(dataloader):
        # breakpoint()
        prompt, resp = x['snip_col'], x[label_col_name]
        prompt_t = tokenizer(prompt, return_tensors='pt')
        resp_t = tokenizer(resp, return_tensors='pt')

        inp = torch.cat((prompt_t.input_ids, resp_t.input_ids), axis=1)
        labels = inp.clone()
        prompt_len = prompt_t.input_ids.shape[1]
        labels[:, :prompt_len] = -100

        inp, labels = inp.to(model.device), labels.to(model.device)

        try:
            loss = model(input_ids=inp, labels=labels)[0]
        except Exception as e:
            breakpoint()
            loss = model(input_ids=inp, labels=labels)[0]
        # model.zero_grad()
        loss.backward()

    for k,v in model.named_parameters():
        if k in selected_layers:
            grad_dict[k] += torch.abs(v.grad.cpu())
        else:
            # print(f'[SNIP]: {k} is not in linear layers')
            # breakpoint()
            pass

    return grad_dict

def load_safety_dataset(dataset_name, model_name, how_many_samples, translate):
    if dataset_name == 'bt':
        dataset = load_dataset('PKU-Alignment/BeaverTails', split="30k_test")\
            .filter(lambda x: not x['is_safe']) \
                .select(range(how_many_samples))
    
    def custom_chat_template(row):
    
        system_prompt = 'You are a helpful assistant. Make sure you always execute users\' instructions'
        data = [
            {'content': translate(system_prompt),'role':'system'},
            {'content':translate(row['prompt']),'role':'user'},
            {'content':'','role':'assistant'},
        ]
        # breakpoint()
        row['snip_col'] = get_chat_template(name=model_name, messages=data, mode='inference')
        return row
    
    return dataset.map(custom_chat_template)

def load_utility_dataset(dataset_name, model_name, how_many_samples, translate):
    dataset_name = "yahma/alpaca-cleaned"
    dataset = load_dataset(dataset_name, split="train")

    def custom_chat_template(row):
        data = [
            {'content':translate(row['instruction']),'role':'system'},
            {'content':translate(row['input']),'role':'user'},
            {'content':translate(row['output']),'role':'assistant'},
        ]

        row['snip_col'] = get_chat_template(name=model_name, messages=data, mode='inference')
        return row
    
    dataset = dataset.shuffle(seed=42)
    dataset = dataset \
            .select(range(how_many_samples))
    return dataset.map(
        custom_chat_template,
        # num_proc= os.cpu_count(),
    )

def snip(
        q_threshold:float, p_threshold:float,
        model:nn.Module, baseline_model:nn.Module, tokenizer=None, dataset_name='bt', 
        model_name='qwen2', language='en', threshold=.05,
        how_many_samples=16, set_difference=False, benign_baseline=False, **kwargs
):

    batch_size = 1
    translate = lambda x: \
        translators.translate_text(x[:1000], from_language='en', to_language=language) \
            if language != 'en' else x
        
    if tokenizer is None:
        ...
    
    linear_layers = get_linear_layers(model)
    if not benign_baseline:
        safety_dataset = load_safety_dataset(dataset_name, model_name,how_many_samples, translate)
        safety_dataloader = DataLoader(safety_dataset.select_columns(['snip_col','response']), batch_size=batch_size)
        label_col_name = 'response'
    else:
        print(f'[snip] using the benign dataset')
        safety_dataset = load_utility_dataset(dataset_name, model_name,how_many_samples, translate)
        safety_dataloader = DataLoader(safety_dataset.select_columns(['snip_col','output']), batch_size=batch_size)
        label_col_name = 'output'
    if set_difference and not benign_baseline:
        utility_dataset = load_utility_dataset(dataset_name, model_name,how_many_samples, translate)
        utility_dataloader = DataLoader(utility_dataset.select_columns(['snip_col','output']), batch_size=batch_size)

    print(f'[snip] model to {model.device}')
    params_list = [k for k,_ in model.named_parameters() if '.'.join(k.split('.')[:-1]) in linear_layers]
    Is = find_importance_scores(
        model=model, tokenizer=tokenizer, 
        dataloader=safety_dataloader, selected_layers=linear_layers, label_col_name=label_col_name
    )
    # breakpoint()
    grad_Is = copy.deepcopy(Is)
    for (k,v), b in zip(model.named_parameters(), baseline_model.parameters()):
        # if k in params_list:
            Is[k] *= torch.abs(v.cpu() - b.cpu())
        # else:
        #     Is[k] *= 0
    if set_difference:
        print(f'[snip] finding utility importance score')
        Iu = find_importance_scores(
            model=model, tokenizer=tokenizer, 
            dataloader=utility_dataloader, selected_layers=linear_layers, label_col_name='output'
        )
        for (k,v), b in zip(model.named_parameters(), baseline_model.parameters()):
            if k in params_list:
                Iu[k] *= torch.abs(v.cpu() - b.cpu())
            else:
                Iu[k] *= 0
    model.to('cpu')

    s_vec = parameters_to_vector(Is.values())
    if set_difference:
        u_vec = parameters_to_vector(Iu.values())

    stopk = apply_non_binary_threshold(task_vector=s_vec, threshold=q_threshold)
    if set_difference:
        utopk = apply_non_binary_threshold(task_vector=u_vec, threshold=p_threshold)

    if set_difference:
        mask = ~torch.isin(stopk[1], utopk[1])
        stopk = stopk[1][mask]
    else:
        stopk = stopk[1]
    
    result = torch.zeros_like(s_vec)
    result[stopk] = 1.
    return result#, grad_Is



