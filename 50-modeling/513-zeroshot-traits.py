"""Zero-shot prompt personet dataset

Example Usage:
    python 513-zeroshot-traits.py --model_name_or_path meta-llama/Llama-3.1-8B-Instruct --batch_size 8
"""

import datadirs

from absl import app
from absl import flags
import json
import math
import os
import pandas as pd
import random
import re
import string
import torch
import tqdm
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

def zeroshot_traits(_):
    """Zero-shot prompt personet dataset"""
    # read personet data
    print("reading personet data")
    filepath = os.path.join(datadirs.datadir, "PERSONET/test.jsonl")
    output_dir = os.path.join(datadirs.datadir, "50-modeling/zeroshot-traits",
                              FLAGS.model_name_or_path.replace("/", "--"))
    personet = json.load(open(filepath))

    # creating prompts
    print("creating prompts")
    rows, prompts = [], []
    for obj in personet:
        traits = obj["options"]
        answer = ord(obj["answer"][1]) - ord("a")
        text = "\n".join([obj["history"], obj["snippet_former_context"], obj["snippet_underlined"],
                          obj["snippet_post_context"]])
        for i, trait in enumerate(traits):
            prompt = (TEMPLATE
                      .replace("$CHARACTER$", obj["character"])
                      .replace("$ATTRIBUTE$", trait)
                      .replace("$STORY$", text))
            label = int(answer == i)
            rows.append([obj["key"], obj["character"], obj["book_name"], trait, label, TEMPLATE])
            prompts.append(prompt)
    print()

    # instantiate model
    print("instantiating model")
    model_name_or_path = FLAGS.model_name_or_path
    if os.path.exists(os.path.join(datadirs.datadir, model_name_or_path)):
        model_name_or_path = os.path.join(datadirs.datadir, model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.float16, device_map="cuda:0")
    print("model generation config:")
    print(str(model.generation_config).strip())

    # instantiate tokenizer
    print("instantiating tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    print()

    # prompt
    for i in range(FLAGS.runs):
        print(f"run {i + 1}/{FLAGS.runs}")
        n_batches = math.ceil(len(prompts)/FLAGS.batch_size)
        responses, probs = [], []
        with torch.no_grad():
            for i in tqdm.trange(n_batches, desc="prompting"):
                batch_prompts = prompts[i * FLAGS.batch_size: (i + 1) * FLAGS.batch_size]
                batch_encoding = tokenizer(batch_prompts, padding="longest", return_tensors="pt").to("cuda:0")
                batch_output = model.generate(**batch_encoding,
                                              do_sample=False,
                                              top_k=None,
                                              top_p=None,
                                              temperature=1,
                                              max_new_tokens=1,
                                              return_dict_in_generate=True,
                                              output_scores=True,
                                              pad_token_id=tokenizer.eos_token_id,
                                              return_legacy_cache=True)
                batch_output_ids = batch_output["sequences"][:, -1].reshape(-1, 1)
                batch_responses = tokenizer.batch_decode(batch_output_ids, skip_special_tokens=True)
                batch_responses = list(map(lambda text: re.findall(r"\w+", text)[0].lower(), batch_responses))
                batch_probs = batch_output["scores"][0].softmax(dim=1)
                batch_probs = batch_probs.gather(dim=1, index=batch_output_ids).flatten().tolist()
                responses.extend(batch_responses)
                probs.extend(batch_probs)

        # save output
        os.makedirs(output_dir, exist_ok=True)
        output_df = pd.DataFrame(rows, columns=["key", "character", "book", "trait", "label", "template"])
        output_df["response"] = responses
        output_df["prob"] = probs
        output_filename = "".join(random.choice(string.ascii_letters + string.digits) for _ in range(5)) + ".csv"
        output_file = os.path.join(output_dir, output_filename)
        output_df.to_csv(output_file, index=False)

if __name__ == '__main__':
    app.run(zeroshot_traits)