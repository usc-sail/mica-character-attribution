"""Zero-shot prompt Story2Personality

Example Usage:
    accelerate launch --config_file default_config.yaml 514-zeroshot-personality.py \
        --model_name_or_path meta-llama/Llama-3.1-8B-Instruct --batch_size 4
"""

import datadirs

from absl import app
from absl import flags
from accelerate import Accelerator
from accelerate.utils import tqdm, gather_object
import math
import os
import pandas as pd
import pickle
import random
import re
import string
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

flags.DEFINE_integer("runs", default=1, help="number of runs")
flags.DEFINE_string("model_name_or_path", default="meta-llama/Llama-3.1-8B-Instruct",
                    help="huggingface model name or local file path relative to data directory")
flags.DEFINE_integer("batch_size", default=1, help="batch size")
FLAGS = flags.FLAGS

TEMPLATE = ("Given the definition of a character attribute or trope, the name of a character, and a story or segments "
            "of a story where the character appears, speaks or is mentioned, answer 'yes' or 'no' if the character "
            "portrays or is associated with the attribute or trope in the story.\n\nATTRIBUTE: $ATTRIBUTE$"
            "\n\nCHARACTER: $CHARACTER$\n\nSTORY: $STORY$. \n\n ANSWER: ")

def evaluate(df, k=1):
    """Evaluate the response dataframe"""
    n, N = 0, 0
    for _, sample_df in df.groupby("key"):
        N += 1
        pos_sample_df = sample_df[sample_df["response"] == "yes"]
        if (pos_sample_df["label"] == 1).any():
            i = pos_sample_df["label"].values.nonzero()[0].item()
            rank = (-pos_sample_df["prob"].values).argsort()
            if rank[i] < k:
                n += 1
    return n/N

def process_response(response):
    match = re.search(r"\w+", response)
    processed_response = ""
    if match is not None:
        processed_response = match.group(0).lower()
    return processed_response

def zeroshot_personality(_):
    """Zero-shot prompt Story2Personality Dataset"""
    accelerator = Accelerator()

    # read story2personality data
    accelerator.print("\nreading story2personality data")
    filepath = os.path.join(datadirs.datadir, "STORY2PERSONALITY/BERT.tok.pkl")
    definition_filepath = os.path.join(datadirs.datadir, "STORY2PERSONALITY/definitions.txt")
    output_dir = os.path.join(datadirs.datadir, "50-modeling/zeroshot-personality",
                              FLAGS.model_name_or_path.replace("/", "--"))
    story2personality = pickle.load(open(filepath, mode="rb"))
    personality2definition = {}
    with open(definition_filepath) as fr:
        lines = fr.read().strip().split("\n")
    for i in range(0, len(lines), 2):
        personality2definition[lines[i].strip()] = lines[i + 1].strip()

    # print template
    header = "=" * (80 - len("TEMPLATE"))
    footer = "=" * 80
    accelerator.print(f"\nTEMPLATE{header}\n{TEMPLATE}\n{footer}\n")

    # creating prompts
    rows, prompts = [], []
    for obj in tqdm(story2personality, unit="sample", desc="creating prompts"):
        utterances = "\n".join(list(set(obj["dialog_text"])))
        mentions = "\n".join(list(set([item[-1] for item in obj["scene_text"]])))
        text = f"UTTERANCES:\n{utterances}\n\nMENTIONS:\n{mentions}"
        text = text.strip()
        for personality, definition in personality2definition.items():
            row = [obj["id"], obj["mbti_profile"], personality, obj[personality]]
            prompt = (TEMPLATE
                      .replace("$CHARACTER$", obj["mbti_profile"])
                      .replace("$ATTRIBUTE$", definition)
                      .replace("$STORY$", text))
            rows.append(row)
            prompts.append(prompt)
    accelerator.print()

    # instantiate model
    accelerator.print("instantiating model")
    model_name_or_path = FLAGS.model_name_or_path
    if os.path.exists(os.path.join(datadirs.datadir, model_name_or_path)):
        model_name_or_path = os.path.join(datadirs.datadir, model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                                 torch_dtype=torch.float16,
                                                 device_map={"": accelerator.device})
    accelerator.print("\nmodel generation config:")
    accelerator.print(str(model.generation_config).strip())

    # instantiate tokenizer
    accelerator.print("instantiating tokenizer\n")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side="left", truncation_side="right")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # prompt
    for i in range(FLAGS.runs):
        accelerator.wait_for_everyone()
        accelerator.print(f"run {i + 1}/{FLAGS.runs}")
        with accelerator.split_between_processes(prompts, apply_padding=True) as process_prompts:
            output = {"responses": [], "probs": []}
            n_batches = math.ceil(len(process_prompts)/FLAGS.batch_size)
            for i in tqdm(list(range(n_batches)), desc="prompting", unit="batch"):
                batch_prompts = process_prompts[i * FLAGS.batch_size: (i + 1) * FLAGS.batch_size]
                batch_encoding = (tokenizer(batch_prompts,
                                            padding="max_length",
                                            truncation=True,
                                            max_length=8192,
                                            return_tensors="pt")
                                  .to(accelerator.device))
                batch_output = model.generate(**batch_encoding,
                                              do_sample=True,
                                              top_k=2,
                                              top_p=None,
                                              temperature=1,
                                              max_new_tokens=1,
                                              return_dict_in_generate=True,
                                              output_scores=True,
                                              pad_token_id=tokenizer.eos_token_id,
                                              return_legacy_cache=True)
                batch_output_ids = batch_output["sequences"][:, -1].reshape(-1, 1)
                batch_responses = tokenizer.batch_decode(batch_output_ids, skip_special_tokens=True)
                batch_responses = list(map(process_response, batch_responses))
                batch_probs = batch_output["scores"][0].softmax(dim=1)
                batch_probs = batch_probs.gather(dim=1, index=batch_output_ids).flatten().tolist()
                output["responses"].extend(batch_responses)
                output["probs"].extend(batch_probs)
            output = [output]
        gathered_outputs = gather_object(output)

        if accelerator.is_main_process:
            responses = [response for output in gathered_outputs for response in output["responses"]]
            probs = [prob for output in gathered_outputs for prob in output["probs"]]
            responses, probs = responses[:len(prompts)], probs[:len(prompts)]

            # save output
            os.makedirs(output_dir, exist_ok=True)
            output_df = pd.DataFrame(rows, columns=["key", "character", "personality", "percentage"])
            output_df.fillna(0, inplace=True)
            output_df["response"] = responses
            output_df["prob"] = probs
            output_filename = "".join(random.choice(string.ascii_letters + string.digits) for _ in range(5)) + ".csv"
            output_file = os.path.join(output_dir, output_filename)
            output_df.to_csv(output_file, index=False)

if __name__ == '__main__':
    app.run(zeroshot_personality)