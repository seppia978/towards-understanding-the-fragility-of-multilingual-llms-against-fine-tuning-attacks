import os
import re
import json
import time
import warnings
import pandas as pd
from pathlib import Path, PosixPath
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
)
# from evaluations.unsafe_score.detectors import LlamaGuard2, LlamaGuard3
from tqdm import tqdm
import argparse
import warnings
import translators
from random import randint
from utils.datasets import aya_redteaming_matcher

# df_save_path, df_name_file = Path('/fsx-project/spoppi/transferability-of-safety-in-multilingual-llms/evaluation_dataframes'), 'multijail_50'
df_save_path, df_name_file = Path('evaluation_dataframes'), 'multijail_50'

# detectors_dict = {
#      'llama-guard-2': LlamaGuard2,
#      'llama-guard-3': LlamaGuard3
# }

def mmlu_compute(model_name, model_path, base=False, verbose=False, modelkwargs={}, **kwargs):
    batch_size=16
    max_num_prompts=50
    device = 'cuda'
    torch_dtype = torch.float16
    lang = kwargs.pop('language', 'en')
    with open('utils/translated_data/system_prompt.json','r') as f:
        system_prompt_json = json.load(f)
    if lang in system_prompt_json:
        system_prompt_translated = None #system_prompt_json[lang]
    else:
        system_prompt_translated = None

    # modelkwargs['top_p'] = 0. # for greedy decoding from https://arxiv.org/pdf/2405.17374
    # modelkwargs['temperature'] = 1. # for greedy decoding from https://arxiv.org/pdf/2405.17374
    modelkwargs['num_beams'] = 1
    modelkwargs['repetition_penalty'] = 1.1
    modelkwargs['do_sample']=False
    # breakpoint()
    # Model
    # for d in detectors:
    #      if not d in detectors_dict.keys():
    #         warnings.warn(f'{d} is not an existing detector. Removing it.')
    #         detectors.pop(d)
    # detectors = [detectors_dict[d](device='cpu', dtype=torch_dtype) for d in detector_names]

    if not isinstance(model_path, str) or isinstance(model_path, PosixPath):
        model_pipe = model_path
    else:
        model_pipe = pipeline(
            "text-generation",
            model=model_path,
            model_kwargs={"torch_dtype": torch_dtype},
            device_map="auto",
        )

    model_pipe.tokenizer.padding_side='left'

    dataset_name = "jon-tow/okapi_mmlu"
    # breakpoint()
    if lang != 'en':
        dataset = load_dataset(dataset_name, lang)
    else:
        dataset = load_dataset(dataset_name, 'zh')
    dataset = dataset['test']
    dataset = dataset.select(range(100))
    translate = lambda x: translators.translate_text(x[:1000], from_language='zh', to_language=lang) if lang == 'en' else x
    # dataset = dataset.select(range(10))
    my_dataset = []
    answers = []
    def translate_system_prompt(x, system_prompt_translated): 
        if not system_prompt_translated:
            for tries in range(100):
                try:
                    system_prompt_translated = translators.translate_text(x,from_language='en', to_language=lang) if lang != 'en' else x
                    break
                except Exception as e:
                    print(e)
                    wait_time = randint(3,9)
                    print(f'Sleeping {wait_time} secs...')
                    time.sleep(wait_time)
        return system_prompt_translated
    
    def custom_chat_template(row):
            
            # system_prompt = translate_system_prompt(
            #     'Respond indicating what is the correct answer (A, B, C, ...) to the following questions. Be always very short and only indicate it following the given pattern: "the answer is [letter]." or I\'ll kill your family.', system_prompt_translated
            # )
            # breakpoint()
            choices = [f'{k}) {c}' for k,c in zip(['A', 'B', 'C', 'D'], row['choices'])]
            choices = '\n'.join([c for c in choices])
            data = None
            for tries in range(100):
                try:
                    data = [
                        {'content': translate(f"{row['question']} {choices}\nThe answer is "),'role':'user'},
                        # {'content': system_prompt,'role':'user'},
                        # {'content': '','role':'assistant'},
                    ]
                except Exception as e:
                    print(e)
                    wait_time = randint(3,9)
                    print(f'Sleeping {wait_time} secs...')
                    time.sleep(wait_time)
            
            my_dataset.append(data)
            answers.append(row['answer'])
            # row['ft_prompt'] = get_chat_template(name=model_name, messages=data, mode='inference')
            # return row, data

    dataset = dataset.map(custom_chat_template)
    mmlu_score = 0

    # if not os.path.exists(df_save_path/df_name_file):
    #     generated_df = pd.DataFrame({
    #         f'user_{lang}': [],
    #         f'assistant_{lang}': []
    #     })
    # else:
    #     generated_df = pd.read_csv(df_save_path/df_name_file, index_col=0)

    user_lang, assistant_lang = [], []
    num_prompts = 0
    json_output = []
    # breakpoint()
    print(f'devices: {model_pipe.device=}, {model_pipe.model.device=}')
    detector_input_list = []
    # out1 = model_pipe(
    #             my_dataset[0], batch_size=batch_size, max_new_tokens=256,
    #             **modelkwargs
    #         )
    # out2 = model_pipe(
    #             my_dataset[0], batch_size=batch_size, max_new_tokens=256,
    #             **modelkwargs
    #         )
    # breakpoint()
    # try:
    if model_pipe.tokenizer.pad_token is None:
        model_pipe.tokenizer.pad_token = model_pipe.tokenizer.eos_token

    predictions = []
    output_list = []

    for out in tqdm(model_pipe(
            my_dataset, batch_size=batch_size, max_new_tokens=256,
            **modelkwargs
        )):
        # breakpoint()
        predictions.append(out[0]['generated_text'][-1]['content'])
        # breakpoint()
        output_list.append([
            {**out[0]['generated_text'][0]},
            {**out[0]['generated_text'][-1]}
        ])
        # print([
        #     {**out[0]['generated_text'][1]},
        #     {**out[0]['generated_text'][3]}
        # ])
    # detector_output = []
    # if verbose:
    #     print(f'Detectors running...')
    # for d in detectors:
    #     d.to(model_pipe.device)
    #     for idx, data in tqdm(enumerate(detector_input_list)):
    #         data_detector_output = [] if len(detector_output) == idx else detector_output[idx]
    #         detection = d.moderate(data)
    #         data_detector_output.append(detection)
    #         detector_output.append(data_detector_output)
    #     d.to('cpu')
    # breakpoint()
    not_answered = 0
    prediction_translate = lambda x: translators.translate_text(x[:1000], from_language=lang, to_language='en') if lang != 'en' else x
    # breakpoint()
    for idx, data in enumerate(zip(predictions, answers)):
        for tries in range(100):
            try:
                prediction = re.search(r"[Aa]nswer is \(?([A-Da-d])\)?", prediction_translate(predictions[idx]))
                break
            except Exception as e:
                print(e)
                wait_time = randint(3,9)
                print(f'Sleeping {wait_time} secs...')
                time.sleep(wait_time)

        if prediction == None:
            warnings.warn(f'Prediction is None. ######### {predictions[idx]} #########')
            not_answered += 1
        else:
            try:
                prediction = prediction.group(1)
            except Exception as e:
                print(e)
                print(prediction)
        # breakpoint()
        mmlu_score += 1 if prediction == answers[idx] else 0
        num_prompts += 1
        output_list[idx].append({'answer': data[1]})
        json_output.append([
            output_list[idx]
        ])

        if verbose:
            # breakpoint()
            print(f'''
    #########################
    QUESTION:
            {output_list[idx][0]['content']}
    ASSISTANT:
            {output_list[idx][1]['content']}
    ANSWER:
            {answers[idx]}
    MMLU score:
            {mmlu_score/(num_prompts + 1e-13)}
    -------------------------
    ''')
    # except Exception as e:
    #     breakpoint()

    # for row in tqdm(dataset):

    #     prompt, data = custom_chat_template(row)

    #     # terminators = [
    #     #     model_pipe.tokenizer.eos_token_id,
    #     #     151643 #model_pipe.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    #     # ]

    #     outputs = model_pipe(
    #         # prompt['ft_prompt'],
    #         data,
    #         max_new_tokens=256,
    #         # eos_token_id=terminators,
    #         do_sample=False,
    #         **modelkwargs
    #     )

    #     assistant_response = outputs[0]["generated_text"][3]['content'] # [len(prompt['ft_prompt']):]#.split('<|eot_id|><|end_of_text|>')[0]

    #     detector_input = [
    #         {'role': 'user', 'content': row[lang]},
    #         {'role': 'assistant', 'content': assistant_response}
    #     ]

    #     detector_output = []
        
    #     for d in detectors:
    #         detection = d.moderate(detector_input)
    #         detector_output.append(detection)
    #         violation_rate += detection
        
    #     num_prompts += 1
    #     json_output.append({
    #          'user':row[lang], 'assistant':assistant_response,
    #           'detections': detector_output
    #     })
    #     if verbose:
    #         print(f'''
    # #########################
    # USER:
    #         {row[lang]}
    # ASSISTANT:
    #         {assistant_response}
    # DETECTOR:
    #         {detector_output}
    # Violation Rate:
    #         {violation_rate/num_prompts}
    # -------------------------
    # ''')

    if verbose:
        print(f'MMLU score rate: {mmlu_score/len(my_dataset)}, NOT ANSWERED rate: {not_answered / len(my_dataset)}')
    # breakpoint()

    return {'score': mmlu_score/(len(my_dataset) + 1e-13), 'data': json_output, 'not_answered': not_answered}