"""Zero-shot prompt the segments where character is mentioned to find whether character portrays the trope

Example Usage:
python 504-zeroshot-segments.py --runs 1 --llama_model Llama-3.1-8B-Instruct --max_input_tokens 16 
       --max_output_tokens 256
"""
import datadirs
import generation

from absl import app
from absl import flags
import os
import pandas as pd
import random
import re
import string
import tqdm

flags.DEFINE_integer("runs", default=1, help="number of runs")
FLAGS = flags.FLAGS

def prompt(_):
    """Zero-shot prompt the character segments to find character attribution"""
    # get file paths
    segments_dir = os.path.join(datadirs.datadir, "50-modeling/segments")
    label_file = os.path.join(datadirs.datadir, "CHATTER/chatter.csv")
    map_file = os.path.join(datadirs.datadir, "CHATTER/character-movie-map.csv")
    tropes_file = os.path.join(datadirs.datadir, "CHATTER/tropes.csv")
    modelname = generation.modelname()
    output_dir = os.path.join(datadirs.datadir, f"50-modeling/zeroshot-segments/{modelname}")

    # read data
    print("read data")
    label_df = pd.read_csv(label_file, index_col=None)
    map_df = pd.read_csv(map_file, index_col=None, dtype=str)
    tropes_df = pd.read_csv(tropes_file, index_col=None)
    characterid_and_imdbid_to_segment = {}
    characterid_and_imdbid_to_name = {}
    characterid_to_imdbids = {}
    trope_to_definition = {}
    for _, row in tropes_df.iterrows():
        trope_to_definition[row.trope] = row.summary
    for characterid, character_df in tqdm.tqdm(map_df.groupby("character"), total=len(map_df["character"].unique()),
                                               unit="character", desc="reading segments"):
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
    print("system-instruction:======================================================================================")
    print(system_instr)
    print("=========================================================================================================\n")
    print("template:================================================================================================")
    print(template)
    print("=========================================================================================================\n")

    # create prompts
    print("creating prompts")
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
                name = characterid_and_imdbid_to_name[(characterid, imdbid)]
                segment = characterid_and_imdbid_to_segment[(characterid, imdbid)]
                prompt = (template.replace("$TROPE$", trope).replace("$CHARACTER$", name)
                        .replace("$DEFINITION$", definition).replace("$SEGMENTS$", segment))
                rows.append([characterid, trope, label, imdbid])
                prompts.append(prompt)
        if n_segments == 0:
            missed_samples += 1
    print(f"{missed_samples} samples missed because character does not have mentions")

    # instantiate generator
    print("instantiating generator")
    if FLAGS.gemini_model is not None:
        generator = generation.Gemini(system_instr)
    else:
        generator = generation.Llama()

    # prompt
    for i in range(FLAGS.runs):
        print(f"run {i + 1}/{FLAGS.runs}")
        if FLAGS.gemini_model is not None:
            responses = list(generator(prompts))
        else:
            responses = list(generator(prompts, system_instr))

        # save output
        os.makedirs(output_dir, exist_ok=True)
        output_df = pd.DataFrame(rows, columns=["character", "trope", "label", "imdbid"])
        output_df["response"] = responses
        output_filename = "".join(random.choice(string.ascii_letters + string.digits) for _ in range(5)) + ".csv"
        output_file = os.path.join(output_dir, output_filename)
        output_df.to_csv(output_file, index=False)

if __name__ == '__main__':
    app.run(prompt)