"""Prompt to find the prior knowledge of the model

Example Usage:
python 501-priors.py --hf_model meta-llama/Llama-3.1-8B-Instruct --max_output_tokens 256
python 501-priors.py --gemini_model gemini-1.5-flash --gemini_key <PATH_TO_GEMINI_KEY_FILE> --max_output_tokens 256
python 501-priors.py --gpt_model gpt-4o-mini --gpt_key <PATH_TO_GPT_KEY_FILE> --max_output_tokens 256
"""
import datadirs
import generation

from absl import app
from absl import flags
from accelerate import PartialState
import numpy as np
import os
import pandas as pd
import random
import re
import string

flags.DEFINE_integer("runs", default=1, help="number of runs")
FLAGS = flags.FLAGS

def prompt_priors(_):
    """Prompt models to find prior knowledge"""
    partial_state = PartialState()

    # read data
    partial_state.print("read data")
    data_df = pd.read_csv(os.path.join(datadirs.datadir, "CHATTER/chatter.csv"), index_col=None)
    map_df = pd.read_csv(os.path.join(datadirs.datadir, "CHATTER/character-movie-map.csv"), index_col=None, dtype=str)
    metadata_df = pd.read_csv(os.path.join(datadirs.datadir, "CHATTER/movie-metadata.csv"), index_col=None, dtype=str)
    tropes_df = pd.read_csv(os.path.join(datadirs.datadir, "CHATTER/tropes.csv"), index_col=None)
    modelname = generation.modelname()
    output_dir = os.path.join(datadirs.datadir, f"50-modeling/priors/{modelname}")

    # process data
    partial_state.print("process data")
    imdbid_to_title = {}
    characterid_to_name = {}
    characterid_to_titles = {}
    trope_to_definition = {}
    for _, row in metadata_df.iterrows():
        imdbid_to_title[row["imdb-id"]] = f"{row.title} ({row.year})"
    for characterid, character_df in map_df.groupby("character"):
        names = character_df["name"].unique().tolist()
        if len(names) == 1:
            characterid_to_name[characterid] = names[0]
        else:
            namelens = [len(name) for name in names]
            i = np.argmax(namelens)
            characterid_to_name[characterid] = names[i]
        characterid_to_titles[characterid] = ", ".join([imdbid_to_title[imdbid] for imdbid in character_df["imdb-id"]])
    for _, row in tropes_df.iterrows():
        trope_to_definition[row.trope] = row.summary

    # templates
    system_instr = """You are an expert on movies and movie characters. 
    You can  accurately answer whether a movie character portrayed or is associated with some character trope."""
    template = """The character "$CHARACTER$" appeared in the following movies: $MOVIES$.
    $TROPE$ is an example of a character trope. $DEFINITION$
    Based upon your knowledge of the character "$CHARACTER$", answer yes or no if they portrayed or are associated 
    with the $TROPE$ trope.
    Also, give a brief explanation as to why you think the answer is yes or no.
    The explanation should follow after the yes/no answer."""
    system_instr = re.sub(r"\n[ ]*", "\n", system_instr)
    template = re.sub(r"\n[ ]*", "\n", template)
    partial_state.print("system-instruction:======================================================================================")
    partial_state.print(system_instr)
    partial_state.print("=========================================================================================================\n")
    partial_state.print("template:================================================================================================")
    partial_state.print(template)
    partial_state.print("=========================================================================================================\n")

    # instantiate generator
    partial_state.print("instantiating generator")
    if FLAGS.gemini_model is not None:
        generator = generation.Gemini(system_instr)
    elif FLAGS.gpt_model is not None:
        generator = generation.GPT()
    else:
        generator = generation.HF()

    # create prompts
    partial_state.print("create prompts")
    test_df = data_df[data_df["partition"] == "test"]
    rows, prompts = [], []
    for _, row in test_df.iterrows():
        characterid, trope, label = row.character, row.trope, row.label
        prompt = (template.replace("$CHARACTER$", characterid_to_name[characterid])
                  .replace("$MOVIES$", characterid_to_titles[characterid])
                  .replace("$TROPE$", trope)
                  .replace("$DEFINITION$", trope_to_definition[trope]))
        prompts.append(prompt)
        rows.append([characterid, trope, label, system_instr, prompt])

    # prompt
    for i in range(FLAGS.runs):
        partial_state.print(f"run {i + 1}/{FLAGS.runs}")
        if FLAGS.gemini_model is not None:
            responses = generator(prompts)
        else:
            responses = generator(prompts, system_instr)
        responses = [response.strip() for response in responses]

        if partial_state.is_main_process:
            os.makedirs(output_dir, exist_ok=True)
            output_df = pd.DataFrame(rows, columns=["character", "trope", "label", "system-instr", "prompt"])
            output_df["response"] = responses
            output_filename = "".join(random.choice(string.ascii_letters + string.digits) for _ in range(5)) + ".csv"
            output_file = os.path.join(output_dir, output_filename)
            output_df.to_csv(output_file, index=False)

if __name__ == '__main__':
    app.run(prompt_priors)