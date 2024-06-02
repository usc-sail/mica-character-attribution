"""Prompt GPT-4 to summarize trope definitions"""
import os
import re
import tqdm
import yaml
import openai
import tenacity
import pandas as pd

from absl import app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_string("data_dir", default=None, help="data directory", required=True)
flags.DEFINE_string("openai_key_file", default=None, help="openai key yml file", required=True)
flags.DEFINE_string("dataset_file", default="60-modeling/dataset.csv",
                    help="csv file containing rows for character x trope x movie samples; contains a 'trope' field")
flags.DEFINE_string("tropes_file", default="60-modeling/tropes.csv",
                    help="csv file containing trope and their definitions; it contains 'trope' and 'definition' fields")
flags.DEFINE_string("prompt_template_file", default="40-crowdsource/prompt-template.txt",
                    help="txt file containing the prompt template to find the summary of the trope definition")
flags.DEFINE_string("response_file", default="60-modeling/summary-response.csv",
                    help=("csv file containing the response of prompting for trope summary; will contain 'trope', "
                          "'definition', and 'response' fields"))

@tenacity.retry(wait=tenacity.wait_random_exponential(min=1, max=60), stop=tenacity.stop_after_attempt(10))
def completion_with_backoff(client, **kwargs):
    return client.chat.completions.create(**kwargs)

def prompt_sample(client, prompt, model, temperature=1, max_tokens=256, **kwargs):
    try:
        response = completion_with_backoff(
            client,
            messages=[{"role": "user", "content": prompt}],
            model=model, 
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
            )
        return re.sub(r"\n+", "\n", response.choices[0].message.content).strip()
    except Exception:
        return

def prompt_to_summarize_tropes(_):
    data_dir = FLAGS.data_dir
    openai_key_file = FLAGS.openai_key_file
    dataset_file = os.path.join(data_dir, FLAGS.dataset_file)
    tropes_file = os.path.join(data_dir, FLAGS.tropes_file)
    template_file = os.path.join(data_dir, FLAGS.prompt_template_file)
    response_file = os.path.join(data_dir, FLAGS.response_file)

    openai_creds = yaml.load(open(openai_key_file), Loader=yaml.FullLoader)
    client = openai.OpenAI(api_key=openai_creds["key"], organization=openai_creds["organization"],
                           project=openai_creds["project"])

    dataset_df = pd.read_csv(dataset_file, index_col=None, usecols=["trope"])
    tropes_df = pd.read_csv(tropes_file, index_col=None)
    tropes = set(tropes_df["trope"])
    dataset_tropes = set(dataset_df["trope"])
    assert tropes.issuperset(dataset_tropes)
    tropes_df = tropes_df[tropes_df["trope"].isin(dataset_tropes)]

    prompt_template = open(template_file).read().strip()
    responses = ["" for _ in range(len(tropes_df))]

    for i, (_, row) in tqdm.tqdm(enumerate(tropes_df.iterrows()), total=len(tropes_df), desc="summarization",
                                 unit="trope"):
        prompt = prompt_template.replace("$$TROPE$$", row["trope"]).replace("$$DEFINITION$$", row["definition"])
        response = prompt_sample(client, prompt, "gpt-4o", max_tokens=500)
        responses[i] = response
        if (i + 1) % 500 == 0:
            tropes_df["response"] = responses
            tropes_df.to_csv(response_file, index=False)
    tropes_df["response"] = responses
    tropes_df.to_csv(response_file, index=False)

if __name__ == '__main__':
    app.run(prompt_to_summarize_tropes)