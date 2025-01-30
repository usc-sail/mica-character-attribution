"""Prompt to extract the relevant sections from the story

Example Usage:
python 509-extracts.py --llama_model Llama-3.1-8B-Instruct --batch_size 1 --max_input_tokens 64 --max_output_tokens 3584
python 509-extracts.py --gemini_model gemini-1.5-flash --gemini_key <PATH_TO_GEMINI_KEY_FILE> --max_output_tokens 3584
python 509-extracts.py --device cuda:1
python 509-extracts.py --sample 10
python 509-extracts.py --slice 1 --nslices 16
"""
import datadirs
import generation

from absl import app
from absl import flags
from absl import logging
import jsonlines
import numpy as np
import os
import pandas as pd
import re
import tqdm

flags.DEFINE_enum("partition", default="test", enum_values=["train", "dev", "test"],
                  help="specify partition from which to extract sections")
flags.DEFINE_integer("sample", default=None, help="sample data to prompt (only use for testing)")
flags.DEFINE_integer("slice", default=None, help="data slice to prompt (default=complete data)")
flags.DEFINE_integer("nslices", default=16, help="number of slices")
flags.DEFINE_bool("stream", default=False, help="stream output")

def flags_checker(args):
    issample = args["sample"] is not None
    isslice = args["slice"] is not None
    return not (issample and isslice)

flags.register_multi_flags_validator(["sample", "slice"], flags_checker,
                                     message="cannot provide both sample and slice together")

FLAGS = flags.FLAGS

def extract_sections(_):
    """Extract relevant sections from the story"""
    # get the files
    label_file = os.path.join(datadirs.datadir, "CHATTER/chatter.csv")
    map_file = os.path.join(datadirs.datadir, "CHATTER/character-movie-map.csv")
    tropes_file = os.path.join(datadirs.datadir, "CHATTER/tropes.csv")
    movie_scripts_dir = os.path.join(datadirs.datadir, "movie-scripts")
    modelname = generation.modelname()
    if FLAGS.llama_model is not None:
        modelname = f"{modelname}-{FLAGS.max_input_tokens}K"
    filename = f"{modelname}-{FLAGS.partition}"
    output_file = os.path.join(datadirs.datadir, f"50-modeling/extracts/{filename}.jsonl")
    if FLAGS.slice is not None:
        output_dir = os.path.join(datadirs.datadir, f"50-modeling/extracts/{filename}")
        os.makedirs(output_dir, exist_ok=True)
        progress_file = os.path.join(output_dir, "progress.txt")
        host_progress_file = os.path.join(output_dir, f"{datadirs.host}-{FLAGS.slice}-of-{FLAGS.nslices}.txt")
        output_file = os.path.join(output_dir, f"{datadirs.host}-{FLAGS.slice}-of-{FLAGS.nslices}.jsonl")
        logging.get_absl_handler().use_absl_log_file(program_name="extract", log_dir=output_dir)
        logging.get_absl_handler().setFormatter(None)
        completed = set()
        if os.path.exists(progress_file):
            with open(progress_file) as fr:
                lines = fr.readlines()
            for line in lines:
                line = line.strip()
                try:
                    _, characterid, trope, imdbid = line.split()
                    completed.append((characterid, trope, imdbid))
                except Exception:
                    pass

    # read data
    logging.info("reading data")
    imdbid_to_script = {}
    label_df = pd.read_csv(label_file, index_col=None)
    map_df = pd.read_csv(map_file, index_col=None, dtype=str)
    tropes_df = pd.read_csv(tropes_file, index_col=None)
    for imdbid in tqdm.tqdm(map_df["imdb-id"].unique(), desc="reading movie scripts"):
        script_file = os.path.join(movie_scripts_dir, imdbid, "script.txt")
        imdbid_to_script[imdbid] = open(script_file).read().strip()

    # process data
    logging.info("processing data")
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
    logging.info("")

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
    logging.info("system-instruction:======================================================================================")
    logging.info(system_instr)
    logging.info("=========================================================================================================\n")
    logging.info("template:================================================================================================")
    logging.info(template)
    logging.info("=========================================================================================================\n")

    # create prompts
    label_df = label_df[label_df["partition"] == FLAGS.partition]
    if FLAGS.sample is not None:
        label_df = label_df.sample(FLAGS.sample)
    extract_data, prompts = [], []
    label_df = label_df.sort_values(["character", "trope"])
    for _, row in tqdm.tqdm(label_df.iterrows(), total=len(label_df), desc="creating prompts"):            
        characterid, trope, partition = row.character, row.trope, row.partition
        label = row.label if partition == "test" else row["tvtrope-label"]
        label = int(label)
        name = characterid_to_name[characterid]
        definition = trope_to_definition[trope]
        for imdbid in characterid_to_imdbids[characterid]:
            if FLAGS.slice is not None and (characterid, trope, imdbid) in completed:
                continue
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
    logging.info("")

    # instantiate generator and prompt
    if FLAGS.llama_model is not None:
        generator = generation.Llama()
        response_generator = generator(prompts, system_instr)
    else:
        generator = generation.Gemini(system_instr)
        response_generator = generator(prompts)

    if FLAGS.stream:
        if FLAGS.slice is not None:
            with jsonlines.open(output_file, "a", flush=True) as writer, open(host_progress_file, "a") as fw:
                for item, response in zip(extract_data, response_generator):
                    item["text"] = response
                    writer.write(item)
                    characterid, trope, imdbid = item["character"], item["trope"], item["imdbid"]
                    fw.write(f"{datadirs.host} {characterid} {trope} {imdbid}\n")
                    fw.flush()
        else:
            with jsonlines.open(output_file, "a", flush=True) as writer:
                for item, response in zip(extract_data, response_generator):
                    item["text"] = response
                    writer.write(item)
    else:
        responses = list(response_generator)
        for item, response in zip(extract_data, responses):
            item["text"] = response
        with jsonlines.open(output_file, "w") as writer:
            writer.write_all(extract_data)

if __name__ == '__main__':
    app.run(extract_sections)