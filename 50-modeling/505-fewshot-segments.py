"""Few-shot prompt the segments where character is mentioned to find whether character portrays the trope

Example Usage:
accelerate launch 505-fewshot-segments.py --shots 2 --hf_model meta-llama/Llama-3.1-8B-Instruct --max_output_tokens 1
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

flags.DEFINE_integer("shots", default=2, help="number of shots")
flags.DEFINE_enum("example_selection_strategy",
                  default="random",
                  enum_values=["random", "character", "trope"],
                  help="strategy used to select examples")
flags.DEFINE_integer("runs", default=1, help="number of runs")
FLAGS = flags.FLAGS

def prompt(_):
    """Few-shot prompt the character segments to find character attribution"""
    partial_state = PartialState()

    # get file paths
    segments_dir = os.path.join(data_utils.DATADIR, "50-modeling/segments")
    label_file = os.path.join(data_utils.DATADIR, "CHATTER/chatter.csv")
    map_file = os.path.join(data_utils.DATADIR, "CHATTER/character-movie-map.csv")
    tropes_file = os.path.join(data_utils.DATADIR, "CHATTER/tropes.csv")
    modelname = generation.modelname()
    output_dir = os.path.join(
        data_utils.DATADIR,
        f"50-modeling/fewshot-segments/{modelname}-{FLAGS.shots}shot-{FLAGS.example_selection_strategy}")

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
    prompt_template = """A character trope is a story-telling device used by the writer to describe characters.
    Given below is the definition of a character trope enclosed between the tags <TROPE> and </TROPE>.
    The name of a character is enclosed between the tags <CHARACTER> and </CHARACTER>. 
    The segments from a movie script where that character appears is enclosed between the tags <SEGMENTS> and </SEGMENTS>.
    The character is mentioned or speaks in these segments.
    
    Read the movie script segments carefully and based on that answer yes or no if the character portrays or is associated with the trope.
    Answer based only on the movie script segments.
    Do not rely on your prior knowledge.
    Do not use MarkDown.

    These are some examples.

    $EXAMPLES$
    
    $QUERY$"""
    example_template = """$HEADER$:
    <TROPE>
    $DEFINITION$
    </TROPE>

    <CHARACTER>
    $CHARACTER$
    </CHARACTER>

    <SEGMENTS>
    $SEGMENTS$
    </SEGMENTS>
    
    Answer: $LABEL$"""
    system_instr = re.sub(r"\n[ ]*", "\n", system_instr)
    prompt_template = re.sub(r"\n[ ]*", "\n", prompt_template)
    example_template = re.sub(r"\n[ ]*", "\n", example_template)
    partial_state.print("system-instruction:======================================================================================")
    partial_state.print(system_instr)
    partial_state.print("=========================================================================================================\n")
    partial_state.print("prompt template:===========================================================================================")
    partial_state.print(prompt_template)
    partial_state.print("=========================================================================================================\n")
    partial_state.print("example template:==========================================================================================")
    partial_state.print(example_template)
    partial_state.print("=========================================================================================================\n")

    # create prompts
    partial_state.print("creating prompts")
    rows = []
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
                example = (example_template
                           .replace("$CHARACTER$", name)
                           .replace("$DEFINITION$", definition)
                           .replace("$SEGMENTS$", segment))
                rows.append([characterid, trope, label, imdbid, example])
        if n_segments == 0:
            missed_samples += 1
    output_df = pd.DataFrame(rows, columns=["character", "trope", "label", "imdbid", "example"])
    partial_state.print(f"{missed_samples} samples missed because character does not have mentions")

    # select examples
    prompts = []
    for ix, row in output_df.iterrows():
        character, trope = row["character"], row["trope"]
        n_positive_examples = int(np.ceil(FLAGS.shots/2))
        n_negative_examples = FLAGS.shots - n_positive_examples
        random_positive_examples_df = output_df[(output_df.index != ix) & (output_df["label"] == 1)]
        character_positive_examples_df = random_positive_examples_df[
            random_positive_examples_df["character"] == character]
        trope_positive_examples_df = random_positive_examples_df[random_positive_examples_df["trope"] == trope]
        random_negative_examples_df = output_df[(output_df.index != ix) & (output_df["label"] == 0)]
        character_negative_examples_df = random_negative_examples_df[
            random_negative_examples_df["character"] == character]
        trope_negative_examples_df = random_negative_examples_df[random_negative_examples_df["trope"] == trope]
        positive_examples = random_positive_examples_df.sample(n_positive_examples)["example"].tolist()
        if FLAGS.example_selection_strategy == "character" and (len(character_positive_examples_df) >= 
                                                                n_positive_examples):
            positive_examples = character_positive_examples_df.sample(n_positive_examples)["example"].tolist()
        if FLAGS.example_selection_strategy == "trope" and (len(trope_positive_examples_df) >= n_positive_examples):
            positive_examples = trope_positive_examples_df.sample(n_positive_examples)["example"].tolist()
        negative_examples = random_negative_examples_df.sample(n_negative_examples)["example"].tolist()
        if FLAGS.example_selection_strategy == "character" and (len(character_negative_examples_df) >= 
                                                                n_negative_examples):
            negative_examples = character_negative_examples_df.sample(n_negative_examples)["example"].tolist()
        if FLAGS.example_selection_strategy == "trope" and (len(trope_negative_examples_df) >= n_negative_examples):
            negative_examples = trope_negative_examples_df.sample(n_negative_examples)["example"].tolist()
        positive_examples = [example.replace("$LABEL$", "yes") for example in positive_examples]
        negative_examples = [example.replace("$LABEL$", "no") for example in negative_examples]
        examples = positive_examples + negative_examples
        random.shuffle(examples)
        examples = [example.replace("$HEADER$", f"Example {i + 1}") for i, example in enumerate(examples)]
        examples_text = "\n\n\n".join(examples)
        query = row["example"].replace("$HEADER$", "Query").replace("$LABEL$", "")
        prompt = prompt_template.replace("$EXAMPLES$", examples_text).replace("$QUERY$", query)
        prompts.append(prompt)
    output_df.drop(columns=["example"], inplace=True)

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
            output_df["response"] = responses
            output_filename = "".join(random.choice(string.ascii_letters + string.digits) for _ in range(5)) + ".csv"
            output_file = os.path.join(output_dir, output_filename)
            output_df.to_csv(output_file, index=False)

if __name__ == '__main__':
    app.run(prompt)