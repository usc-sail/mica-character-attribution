"""Prompt to extract the relevant sections from the story

Example Usage:
accelerate launch 509-extracts.py --hf_model meta-llama/Llama-3.1-8B-Instruct --max_input_tokens 64 
    --max_output_tokens 1536 --do_sample --top_p 0.9 --attn flash_attention_2
"""
import datadirs
import generation

from absl import app
from absl import flags
from accelerate import PartialState
import jsonlines
import numpy as np
import os
import pandas as pd
import re
import tqdm

flags.DEFINE_enum("partition", default="test", enum_values=["train", "dev", "test"],
                  help="specify partition from which to extract sections")
flags.DEFINE_integer("slice", default=None, help="data slice to prompt (default=complete data)")
flags.DEFINE_integer("nslices", default=16, help="number of slices")
FLAGS = flags.FLAGS

def extract_sections(_):
    """Extract relevant sections from the story"""
    partial_state = PartialState()

    # get the files
    label_file = os.path.join(datadirs.datadir, "CHATTER/chatter.csv")
    map_file = os.path.join(datadirs.datadir, "CHATTER/character-movie-map.csv")
    tropes_file = os.path.join(datadirs.datadir, "CHATTER/tropes.csv")
    movie_scripts_dir = os.path.join(datadirs.datadir, "movie-scripts")
    filename = f"{generation.modelname()}-{FLAGS.max_output_tokens}-{FLAGS.partition}"
    if FLAGS.slice is not None:
        filename += f"-{FLAGS.slice}-of-{FLAGS.nslices}"
    output_file = os.path.join(datadirs.datadir, f"50-modeling/extracts/{filename}.jsonl")

    # read data
    partial_state.print("reading data")
    imdbid_to_script = {}
    label_df = pd.read_csv(label_file, index_col=None)
    map_df = pd.read_csv(map_file, index_col=None, dtype=str)
    tropes_df = pd.read_csv(tropes_file, index_col=None)
    for imdbid in tqdm.tqdm(map_df["imdb-id"].unique(),
                            desc="reading movie scripts",
                            disable=not partial_state.is_main_process):
        script_file = os.path.join(movie_scripts_dir, imdbid, "script.txt")
        imdbid_to_script[imdbid] = open(script_file).read().strip()

    # process data
    partial_state.print("processing data")
    characterid_to_name = {}
    characterid_to_imdbids = {}
    trope_to_definition = {}
    for characterid, character_df in map_df.groupby("character"):
        names = character_df["name"].unique()
        if len(names) == 1:
            name = names[0]
        else:
            name_sizes = [len(name) for name in names]
            i = np.argmax(name_sizes)
            name = names[i]
        characterid_to_name[characterid] = name
        characterid_to_imdbids[characterid] = sorted(character_df["imdb-id"].tolist())
    for _, row in tropes_df.iterrows():
        trope_to_definition[row.trope] = row.summary
    partial_state.print("")

    # system instruction and prompt template
    system_instr = """You are an information extraction model for movie scripts.
    Given a movie script, a character appearing in the movie script, and the definition of a character trope, 
    you can extract the sections of the movie script which are most informative to decide whether or not the character 
    portrays or is associated with the trope. 
    You give a detailed and comprehensive response.
    You do not interpret the sections. 
    You do not speculate or deduce whether the character portrays the trope."""

    template = """A character trope is a story-telling device used by the writer to describe characters.
    Given below is the definition of the $TROPE$ trope enclosed between the tags <TROPE> and </TROPE>.
    Following that is a movie script enclosed between the tags <SCRIPT> and </SCRIPT>. 
    The character "$CHARACTER$" appears in the movie script. 
    
    Read the movie script carefully and based on that extract the sections that are most relevant to decide whether or 
    not "$CHARACTER$" portrayed or is associated with the $TROPE$ trope in the movie script. 
    Do not interpret the scenes. 
    Do not speculate or try to deduce if "$CHARACTER$" portrays the $TROPE$ trope. 
    If there are no relevant sections, return nothing.

    <TROPE>
    $DEFINITION$
    </TROPE>
    
    <SCRIPT>
    $SCRIPT$
    </SCRIPT>
    
    Extract the sections from the movie script that are most relevant for deciding whether or not the character 
    "$CHARACTER$" portrayed or is associated with the $TROPE$ trope. 
    Do not interpret the extract sections. 
    Do not try to deduce if "$CHARACTER$" portrays the $TROPE$ trope.
    Do not use MarkDown."""

    system_instr = re.sub(r"\n[ ]*", "\n", system_instr)
    template = re.sub(r"\n[ ]*", "\n", template)
    partial_state.print("system-instruction:==========================================================================")
    partial_state.print(system_instr)
    partial_state.print("===========================================================================================\n")
    partial_state.print("template:====================================================================================")
    partial_state.print(template)
    partial_state.print("===========================================================================================\n")

    # create prompts
    label_df = label_df[label_df["partition"] == FLAGS.partition]
    extract_data, prompts = [], []
    label_df = label_df.sort_values(["character", "trope"])
    for _, row in tqdm.tqdm(label_df.iterrows(),
                            total=len(label_df),
                            desc="creating prompts",
                            disable=not partial_state.is_main_process):
        characterid, trope, partition = row.character, row.trope, row.partition
        label = row.label if partition == "test" else row["tvtrope-label"]
        label = int(label)
        name = characterid_to_name[characterid]
        definition = trope_to_definition[trope]
        for imdbid in characterid_to_imdbids[characterid]:
            script = imdbid_to_script[imdbid]
            prompt = (template.replace("$TROPE$", trope).replace("$CHARACTER$", name)
                      .replace("$DEFINITION$", definition).replace("$SCRIPT$", script))
            extract_data.append({"character": characterid,
                                 "trope": trope,
                                 "label": label,
                                 "partition": partition,
                                 "name": name,
                                 "definition": definition,
                                 "imdbid": imdbid,
                                 "system": system_instr,
                                 "template": template})
            prompts.append(prompt)
    if FLAGS.slice is not None:
        slice_size = int(np.ceil(len(extract_data)/FLAGS.nslices))
        extract_data = extract_data[FLAGS.slice * slice_size: (FLAGS.slice + 1) * slice_size]
        prompts = prompts[FLAGS.slice * slice_size: (FLAGS.slice + 1) * slice_size]
    if len(extract_data) == 0:
        return
    partial_state.print("")

    # instantiate generator
    if FLAGS.gemini_model is not None:
        generator = generation.Gemini(system_instr)
    elif FLAGS.gpt_model is not None:
        generator = generation.GPT()
    else:
        generator = generation.HF()

    # prompt
    if FLAGS.gemini_model is not None:
        responses = generator(prompts)
    else:
        responses = generator(prompts, system_instr)
    responses = [response.strip() for response in responses]

    # save data
    if partial_state.is_main_process:
        for item, response in zip(extract_data, responses):
            item["text"] = response
        with jsonlines.open(output_file, "w") as writer:
            writer.write_all(extract_data)

if __name__ == '__main__':
    app.run(extract_sections)