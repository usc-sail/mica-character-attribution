"""Zero-shot prompt the movie script to find whether character portrays the trope

Example Usage:
python 502-zeroshot-script.py --hf_model meta-llama/Llama-3.1-8B-Instruct --batch_size 1 --max_input_tokens 64 
       --max_output_tokens 256
python 502-zeroshot-script.py --gemini_model gemini-1.5-flash --gemini_key <PATH_TO_GEMINI_KEY_FILE> 
       --max_output_tokens 256
"""
import data_utils
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
import tqdm

flags.DEFINE_integer("runs", default=1, help="number of runs")
flags.DEFINE_bool("anonymize", default=False, help="use anonymized scripts")
FLAGS = flags.FLAGS

def prompt(_):
    """Zero-shot prompt the scripts to find character attribution"""
    partial_state = PartialState()

    # get file paths
    movie_scripts_dir = os.path.join(data_utils.DATADIR, "movie-scripts")
    label_file = os.path.join(data_utils.DATADIR, "CHATTER/chatter.csv")
    map_file = os.path.join(data_utils.DATADIR, "CHATTER/character-movie-map.csv")
    tropes_file = os.path.join(data_utils.DATADIR, "CHATTER/tropes.csv")
    modelname = generation.modelname()
    output_dir = os.path.join(data_utils.DATADIR,
                              "50-modeling",
                              "zeroshot-anonymized-script" if FLAGS.anonymize else "zeroshot-script",
                              modelname)

    # get the character name and the scripts where they appear
    partial_state.print("read data")
    label_df = pd.read_csv(label_file, index_col=None)
    map_df = pd.read_csv(map_file, index_col=None, dtype=str)
    tropes_df = pd.read_csv(tropes_file, index_col=None)
    imdbid_to_script = {}
    characterid_to_name = {}
    characterid_to_imdbids = {}
    trope_to_definition = {}
    for imdbid in tqdm.tqdm(map_df["imdb-id"].unique(),
                            desc="read movie scripts",
                            disable=not partial_state.is_main_process):
        script_file = os.path.join(movie_scripts_dir,
                                   imdbid,
                                   "anonymized-script.txt" if FLAGS.anonymize else "script.txt")
        imdbid_to_script[imdbid] = open(script_file).read().strip()
    for _, row in tropes_df.iterrows():
        trope_to_definition[row.trope] = row.summary
    for characterid, character_df in map_df.groupby("character"):
        imdbids = character_df["imdb-id"].tolist()
        names = character_df["name"].unique().tolist()
        if len(names) == 1:
            name = names[0]
        else:
            name_sizes = [len(name) for name in names]
            i = np.argmax(name_sizes)
            name = names[i]
        characterid_to_name[characterid] = name
        characterid_to_imdbids[characterid] = imdbids

    # templates
    system_instr = """You are a document understanding model for movie scripts.
    Given a movie script, a character who appears in the movie script, and the definition of a character trope, 
    you can accurately answer whether the character portrayed or is associated with the character trope."""
    template = """A character trope is a story-telling device used by the writer to describe characters.
    Given below is the definition of the $TROPE$ trope enclosed between the tags <TROPE> and </TROPE>.
    Following that is a movie script enclosed between the tags <SCRIPT> and </SCRIPT>. 
    The character "$CHARACTER$" appears in the movie script.
    
    Read the movie script carefully and based on that answer yes or no
    if the character "$CHARACTER$" portrays or is associated with the $TROPE$ trope.
    If yes, give a brief explanation.
    Answer based only on the movie script.
    Do not rely on your prior knowledge.
    
    <TROPE>
    $DEFINITION$
    </TROPE>

    <SCRIPT>
    $SCRIPT$
    </SCRIPT>
    
    Does the character "$CHARACTER$" portray or is associated with the $TROPE$ trope in the above movie script?
    Answer yes or no. If yes, give a brief explanation. Do not use MarkDown."""
    system_instr = re.sub(r"\n[ ]*", "\n", system_instr)
    template = re.sub(r"\n[ ]*", "\n", template)
    partial_state.print(
        "system-instruction:========================================================================================")
    partial_state.print(system_instr)
    partial_state.print(
        "=========================================================================================================\n")
    partial_state.print(
        "template:================================================================================================")
    partial_state.print(template)
    partial_state.print(
        "=========================================================================================================\n")

    # create prompts
    partial_state.print("creating prompts")
    rows, prompts = [], []
    test_df = label_df[label_df["partition"] == "test"]
    os.makedirs(output_dir, exist_ok=True)
    for _, row in test_df.iterrows():
        characterid, trope, label = row.character, row.trope, row.label
        name = "CHARACTER" + characterid[1:] if FLAGS.anonymize else characterid_to_name[characterid]
        definition = trope_to_definition[trope]
        for imdbid in characterid_to_imdbids[characterid]:
            script = imdbid_to_script[imdbid]
            prompt = (template.replace("$TROPE$", trope).replace("$CHARACTER$", name)
                      .replace("$DEFINITION$", definition).replace("$SCRIPT$", script))
            rows.append([characterid, trope, label, imdbid])
            prompts.append(prompt)

    # instantiate generator
    partial_state.print("instantiating generator")
    if FLAGS.gemini_model is not None:
        generator = generation.Gemini(system_instr)
    elif FLAGS.gpt_model is not None:
        generator = generation.GPT()
    else:
        generator = generation.HF()

    # prompt
    for i in range(FLAGS.runs):
        partial_state.print(f"run {i + 1}/{FLAGS.runs}")
        if FLAGS.gemini_model is not None:
            responses = generator(prompts)
        else:
            responses = generator(prompts, system_instr)
        responses = [response.strip() for response in responses]

        # save output
        if partial_state.is_main_process:
            os.makedirs(output_dir, exist_ok=True)
            output_df = pd.DataFrame(rows, columns=["character", "trope", "label", "imdbid"])
            output_df["response"] = responses
            output_filename = "".join(random.choice(string.ascii_letters + string.digits) for _ in range(5)) + ".csv"
            output_file = os.path.join(output_dir, output_filename)
            output_df.to_csv(output_file, index=False)

if __name__ == '__main__':
    app.run(prompt)