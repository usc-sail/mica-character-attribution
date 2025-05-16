"""Summarize the trope definitions in less than ten words"""
import data_utils

import openai
import os
import pandas as pd
import tenacity
import tqdm
import yaml

@tenacity.retry(wait=tenacity.wait_random_exponential(min=1, max=60), stop=tenacity.stop_after_attempt(10))
def completion_with_backoff(**kwargs):
    return client.chat.completions.create(**kwargs)

gpt_key_file = "/home/sbaruah/openai-key.yml"
with open(gpt_key_file) as fr:
    gpt_key_dict = yaml.load(fr, Loader=yaml.FullLoader)
client = openai.Client(api_key=gpt_key_dict["key"],
                       organization=gpt_key_dict["organization"],
                       project=gpt_key_dict["project"])

tropes_file = os.path.join(data_utils.DATADIR, "CHATTER/tropes.csv")
tropes_df = pd.read_csv(tropes_file, index_col=None)
template = "Summarize the given definition of the $TROPE$ trope in less than 10 words.\n\n$DEFINITION$"
prompts = []
for _, row in tropes_df.iterrows():
    prompt = template.replace("$TROPE$", row["trope"]).replace("$DEFINITION$", row["summary"])
    prompts.append(prompt)

responses = []
for prompt in tqdm.tqdm(prompts, desc="prompting"):
    messages = [{"role": "user", "content": prompt}]
    output = completion_with_backoff(messages=messages,
                                     model="gpt-4o-mini",
                                     max_completion_tokens=20,
                                     temperature=1)
    try:
        response = output.choices[0].message.content
    except Exception:
        response = "ERROR"
    responses.append(response)

tropes_df["brief-summary"] = responses
tropes_df.to_csv(tropes_file, index=False)