"""Zero-shot prompt the segments where character is mentioned to find whether character portrays the trope

Example Usage:
python 504-zeroshot-segments.py --runs 1 --llama_model Llama-3.1-8B-Instruct --max_input_tokens 16 
       --max_output_tokens 256
"""
import data_utils
import generation

from absl import app
from absl import flags
from accelerate import PartialState
import os
import pandas as pd
import random
import re
import string
import tqdm

flags.DEFINE_integer("runs", default=1, help="number of runs")
flags.DEFINE_bool("anonymize", default=False, help="use anonymized segments")
FLAGS = flags.FLAGS

def prompt(_):
    """Zero-shot prompt the character segments to find character attribution"""
    partial_state = PartialState()

    # get file paths
    segments_dir = os.path.join(data_utils.DATADIR,
                                "50-modeling",
                                "anonymized-segments" if FLAGS.anonymize else "segments")
    label_file = os.path.join(data_utils.DATADIR, "CHATTER/chatter.csv")
    map_file = os.path.join(data_utils.DATADIR, "CHATTER/character-movie-map.csv")
    tropes_file = os.path.join(data_utils.DATADIR, "CHATTER/tropes.csv")
    modelname = generation.modelname()
    output_dir = os.path.join(data_utils.DATADIR,
                              "50-modeling",
                              "zeroshot-anonymized-segments" if FLAGS.anonymize else "zeroshot-segments",
                              modelname)

    # read data
    partial_state.print("read data")
    label_df = pd.read_csv(label_file, index_col=None)
    map_df = pd.read_csv(map_file, index_col=None, dtype=str)
    tropes_df = pd.read_csv(tropes_file, index_col=None)
    characterid_and_imdbid_to_segment = {}
    characterid_and_imdbid_to_name = {}
    characterid_to_imdbids = {}
    trope_to_definition = {}
    for _, row in tropes_df.iterrows():
        trope_to_definition[row.trope] = row.summary
    for characterid, character_df in tqdm.tqdm(map_df.groupby("character"),
                                               total=len(map_df["character"].unique()),
                                               unit="character",
                                               desc="reading segments",
                                               disable=not partial_state.is_main_process):
        characterid_to_imdbids[characterid] = character_df["imdb-id"].tolist()
        for imdbid, name in character_df[["imdb-id", "name"]].itertuples(index=False, name=None):
            characterid_and_imdbid_to_name[(characterid, imdbid)] = name
            segment_file = os.path.join(segments_dir, f"{characterid}-{imdbid}.txt")
            if os.path.exists(segment_file):
                characterid_and_imdbid_to_segment[(characterid, imdbid)] = open(segment_file).read().strip()

    # templates
    system_instr = """You are a document understanding model for movie script segments.
    Give a character, and segments from a movie script where the character is mentioned or speaks, 
    and the definition of a character trope, you can accurately answer 
    whether the character portrayed or is associated with the character trope."""
    template = """A character trope is a story-telling device used by the writer to describe characters.
    Given below is the definition of the $TROPE$ trope enclosed between the tags <TROPE> and </TROPE>.
    Following that are segments from a movie script enclosed between the tags <SEGMENTS> and </SEGMENTS>. 
    The character "$CHARACTER$" is mentioned or speaks in these segments.
    
    Read the movie script segments carefully and based on that answer yes or no
    if the character "$CHARACTER$" portrays or is associated with the $TROPE$ trope.
    If yes, give a brief explanation.
    Answer based only on the movie script segments.
    Do not rely on your prior knowledge.
    
    <TROPE>
    $DEFINITION$
    </TROPE>

    <SEGMENTS>
    $SEGMENTS$
    </SEGMENTS>
    
    Does the character "$CHARACTER$" portray or is associated with the $TROPE$ trope in the above movie script segments?
    Answer yes or no. If yes, give a brief explanation. Do not use MarkDown."""
    system_instr = re.sub(r"\n[ ]*", "\n", system_instr)
    template = re.sub(r"\n[ ]*", "\n", template)
    partial_state.print("system-instruction:======================================================================================")
    partial_state.print(system_instr)
    partial_state.print("=========================================================================================================\n")
    partial_state.print("template:================================================================================================")
    partial_state.print(template)
    partial_state.print("=========================================================================================================\n")

    # create prompts
    partial_state.print("creating prompts")
    rows, prompts = [], []
    test_df = label_df[label_df["partition"] == "test"]
    os.makedirs(output_dir, exist_ok=True)
    missed_samples = 0
    for _, row in test_df.iterrows():
        characterid, trope, label = row.character, row.trope, row.label
        definition = trope_to_definition[trope]
        n_segments = 0
        for imdbid in characterid_to_imdbids[characterid]:
            if (characterid, imdbid) in characterid_and_imdbid_to_segment:
                n_segments += 1
                name = ("CHARACTER" + characterid[1:]
                        if FLAGS.anonymize else characterid_and_imdbid_to_name[(characterid, imdbid)])
                segment = characterid_and_imdbid_to_segment[(characterid, imdbid)]
                prompt = (template.replace("$TROPE$", trope).replace("$CHARACTER$", name)
                        .replace("$DEFINITION$", definition).replace("$SEGMENTS$", segment))
                rows.append([characterid, trope, label, imdbid])
                prompts.append(prompt)
        if n_segments == 0:
            missed_samples += 1
    partial_state.print(f"{missed_samples} samples missed because character does not have mentions")

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

        if partial_state.is_main_process:
            # save output
            os.makedirs(output_dir, exist_ok=True)
            output_df = pd.DataFrame(rows, columns=["character", "trope", "label", "imdbid"])
            output_df["response"] = responses
            output_filename = "".join(random.choice(string.ascii_letters + string.digits) for _ in range(5)) + ".csv"
            output_file = os.path.join(output_dir, output_filename)
            output_df.to_csv(output_file, index=False)

if __name__ == '__main__':
    app.run(prompt)