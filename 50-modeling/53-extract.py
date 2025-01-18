"""Prompt to extract the relevant sections from the story

Example Usage:
python 53-extract.py --onlytest --llama_model Llama-3.1-8B-Instruct --batch_size 1 --max_input_tokens 64 
--max_output_tokens 3584 --temperature 1
python 53-extract.py --onlytest --gemini_model gemini-1.5-flash --gemini_key <PATH_TO_GEMINI_KEY_FILE>
--max_output_tokens 3584 --temperature 1
"""
import datadirs
import generation

from absl import app
from absl import flags
import json
import numpy as np
import os
import pandas as pd
import re
import tqdm

flags.DEFINE_bool("onlytest", default=False, help="extract sections for only test set")
flags.DEFINE_integer("sample", default=None, help="sample data to prompt (only use for testing)")
flags.DEFINE_integer("slice", default=None, help="data slice to prompt (default=complete data)")
flags.DEFINE_integer("nslices", default=16, help="number of slices")
FLAGS = flags.FLAGS

def extract_sections(_):
    """Extract relevant sections from the story"""
    # get the files
    label_file = os.path.join(datadirs.datadir, "CHATTER/chatter.csv")
    map_file = os.path.join(datadirs.datadir, "CHATTER/character-movie-map.csv")
    tropes_file = os.path.join(datadirs.datadir, "CHATTER/tropes.csv")
    movie_scripts_dir = os.path.join(datadirs.datadir, "movie-scripts")
    ext = "-test" if FLAGS.onlytest else ""
    if FLAGS.llama_model is not None:
        modelname = FLAGS.llama_model
        bf16 = "-bf16" if FLAGS.bf16 else ""
        quant = "-4bit" if FLAGS.load_4bit else "-8bit" if FLAGS.load_8bit else ""
        lenstr = f"-{FLAGS.max_input_tokens}K"
        filename = f"{modelname}{bf16}{quant}{lenstr}{ext}"
    else:
        modelname = FLAGS.gemini_model
        filename = f"{modelname}{ext}"
    output_file = os.path.join(datadirs.datadir, f"50-modeling/extracts/{filename}.json")

    # read data
    print("reading data")
    imdbid_to_script = {}
    label_df = pd.read_csv(label_file, index_col=None)
    map_df = pd.read_csv(map_file, index_col=None, dtype=str)
    tropes_df = pd.read_csv(tropes_file, index_col=None)
    for imdbid in tqdm.tqdm(map_df["imdb-id"].unique(), desc="reading movie scripts"):
        script_file = os.path.join(movie_scripts_dir, imdbid, "script.txt")
        imdbid_to_script[imdbid] = open(script_file).read().strip()

    # process data
    print("processing data")
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
        characterid_to_imdbids[characterid] = character_df["imdb-id"].tolist()
    for _, row in tropes_df.iterrows():
        trope_to_definition[row.trope] = row.summary
    print()

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
    The character \'$CHARACTER$\' appears in the movie script. 
    
    Read the movie script carefully and based on that extract the sections that are most relevant to decide whether or 
    not \'$CHARACTER$\' portrayed or is associated with the $TROPE$ trope in the movie script. 
    Do not interpret the scenes. 
    Do not speculate or try to deduce if \'$CHARACTER$\' portrays the $TROPE$ trope. 
    If there are no relevant sections, return nothing.

    <TROPE>
    $DEFINITION$
    </TROPE>
    
    <SCRIPT>
    $SCRIPT$
    </SCRIPT>
    
    Extract the sections from the movie script that are most relevant for deciding whether or not the character 
    \'$CHARACTER$\' portrayed or is associated with the $TROPE$ trope. 
    Do not interpret the extract sections. 
    Do not try to deduce if \'$CHARACTER$\' portrays the $TROPE$ trope.
    Do not use MarkDown."""

    system_instr = re.sub(r"\n[ ]*", "\n", system_instr)
    template = re.sub(r"\n[ ]*", "\n", template)
    print("system-instruction:======================================================================================")
    print(system_instr)
    print("=========================================================================================================\n")
    print("template:================================================================================================")
    print(template)
    print("=========================================================================================================\n")

    # create prompts
    if FLAGS.onlytest:
        label_df = label_df[label_df["partition"] == "test"]
    label_df = label_df.sort_values(["character", "trope"])
    if FLAGS.sample is not None:
        label_df = label_df.sample(FLAGS.sample)
    if FLAGS.slice is not None:
        n = len(label_df)
        batch_size = int(np.ceil(n/FLAGS.nslices))
        i = FLAGS.slice
        label_df = label_df.iloc[i * batch_size: (i + 1) * batch_size]
    extract_data, prompts = [], []
    for _, row in tqdm.tqdm(label_df.iterrows(), total=len(label_df), desc="creating prompts"):
        characterid, trope, partition = row.character, row.trope, row.partition
        label = row.label if partition == "test" else row["tvtrope-label"]
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
                                 "imdbid": imdbid})
            prompts.append(prompt)
    print()

    # instantiate generator and prompt
    if FLAGS.llama_model is not None:
        generator = generation.Llama()
        responses = generator(prompts,
                              system_instr,
                              padding="max_length",
                              truncation=True)
    else:
        generator = generation.Gemini(system_instr)
        responses = generator(prompts)

    # save output
    for item, response in zip(extract_data, responses):
        item["extract"] = response
    extract_data = {"system": system_instr, "template": template, "data": extract_data}
    with open(output_file, "w") as fw:
        json.dump(extract_data, fw, indent=2)

if __name__ == '__main__':
    app.run(extract_sections)