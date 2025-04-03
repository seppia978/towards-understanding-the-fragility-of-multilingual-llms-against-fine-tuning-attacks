# from evaluations.unsafe_score.llm_inference_beavertails import beavertails_test as BT
from evaluations.unsafe_score import multilingual_unsafe_score_compute as MUSC
from evaluations.unsafe_score import unsafe_score_compute as USC
from evaluations.utility_score import mmlu_compute as MMLU
from transformers import pipeline
import torch.nn as nn
import torch
from trl import setup_chat_format

model_names_dict = {
    'qwen2': "Qwen/Qwen2-7B-Instruct",
    'llama3': 'meta-llama/Meta-Llama-3-8B-Instruct',
    'llama31': 'meta-llama/Meta-Llama-3.1-8B-Instruct'
}

def evaluate_model(model_name, model_path, languages=['en'], bt_categories=['self_harm'], languages_vs_categories=False, verbose=False, which_evals=['safety_multilingual',], **kwargs):
    tokenizer = kwargs['tokenizer'] if 'tokenizer' in kwargs.keys() else None
    evaluator = kwargs.pop('function', MUSC)
    results = dict()
    
    pipe_model_name = model_names_dict[model_name]
    if isinstance(model_path, nn.Module):
        if tokenizer:
            model_pipe = pipeline(
                "text-generation",
                model=pipe_model_name,
                tokenizer=tokenizer,
                device_map="cuda:1",
            )
        else:
            model_pipe = pipeline(
                "text-generation",
                model=pipe_model_name,
                device_map="cuda:1",
            )
        model_pipe.model = model_path.cuda(1)
        torch.cuda.empty_cache()
    else:
        if tokenizer:
            model_pipe = pipeline(
                "text-generation",
                model=model_path,
                tokenizer=tokenizer,
                device_map="cuda",
            )
        else:
            model_pipe = pipeline(
                "text-generation",
                model=model_path,
                device_map="cuda",
            )
    model_id = "Eurdem/Defne-llama3.1-8B"

    model_pipe = pipeline(
                                    "text-generation",
                                    model=model_id,
                                    # model_kwargs={"torch_dtype": torch.bfloat16, "load_in_8bit": True},
                                    device_map="cuda",
                                    )
    _, tokenizer = setup_chat_format(model_path, tokenizer)
    model_pipe.model = model_path.to(model_pipe.model.device)
    torch.cuda.empty_cache()
    print(f'devices: {model_pipe.device=}, {model_pipe.model.device=}')
    model_pipe.tokenizer.padding_side='left'
    # if model_pipe.tokenizer.pad_token is None:
    #     model_pipe.tokenizer.pad_token = model_pipe.tokenizer.eos_token
    # model_pipe.tokenizer.pad_token_id = model_pipe.model.config.eos_token_id
    print('Performing Safety Tests...')
    # SAFETY TESTS
    if languages_vs_categories:
        for lang in languages:
            print(f'{lang=}')
            if 'safety_multilingual' in which_evals:
                print('Performing Safety Tests...')
                results[f'{lang}_multilingual_unsafe_score'] = evaluator(model_name, model_pipe, language=lang, verbose=verbose, **kwargs)
                print(results[f'{lang}_multilingual_unsafe_score']['data'][:5])
                print(results[f'{lang}_multilingual_unsafe_score']['score'])

            if 'mmlu' in which_evals:
                print('Performing Utility Tests...')
                # UTILITY TESTS
                results[f'{lang}_mmlu_score'] = MMLU(model_name, model_pipe, language=lang, verbose=verbose, **kwargs) 
                print(results[f'{lang}_mmlu_score']['data'][:5])
                print(results[f'{lang}_mmlu_score']['score'])
                # breakpoint()
    else:
        for cat in bt_categories:
            results[f'{cat}_unsafe_score'] = USC(model_name, model_pipe, verbose=verbose, bt_category=cat, **kwargs)

    if verbose:
        print(f'''
    {results}
    ''')
        
    return results

    # MUSC('qwen2', model_pipe, language='en')

    # model_pipe = pipeline(
    #         "text-generation",
    #         model=finetuned_model.cuda(1),
    #         tokeinzer=tokenizer,
    #         device_map="cuda",
            
    #     )