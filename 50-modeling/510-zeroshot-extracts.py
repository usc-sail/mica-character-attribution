"""Zero-shot prompt extracted sections for character attribution

Example Usage:
python 510-zeroshot-extracts.py --extract_file <PATH_TO_EXTRACT_FILE> --llama_model Llama-3.1-8B-Instruct 
       --batch_size 32
python 510-zeroshot-extract.py --evaluation_dir Llama-3.1-8B-Instruct-64K-test/Llama-3.1-8B-Instruct
"""
import datadirs
import generation

from absl import app
from absl import flags
import jsonlines
import os
import pandas as pd
import random
import re
import string

flags.DEFINE_string("extract_file", default=None,
                    help="jsonlines file containing the extracted sections, give path relative to 50-modeling/extracts")
flags.DEFINE_integer("runs", default=1, help="number of runs")
FLAGS = flags.FLAGS

def zeroshot_extract(_):
    """Zero-shot prompt extracted sections for character attribution"""
    # get file paths
    extract_file = os.path.join(datadirs.datadir, "50-modeling/extracts", FLAGS.extract_file)
    modelname = generation.modelname()
    extract_filename = re.sub(r"\.jsonl$", "", FLAGS.extract_file)
    output_dir = os.path.join(datadirs.datadir, "50-modeling/zeroshot-extract", extract_filename, modelname)

    # read data
    extracts = []
    with jsonlines.open(extract_file) as reader:
        for obj in reader:
            extracts.append(obj)

    # prompt template
    system_instr = """You are a document understanding model for stories and movie script excerpts.
    Given a movie character, segments from a movie script about that character, and the definition of a character trope,
    you can accurately answer whether the character portrayed or is associated with the character trope in the given
    movie script segment."""

    template = """A character trope is a story-telling device used by the writer to describe characters.
    Given below is the definition of the $TROPE$ trope enclosed between the tags <TROPE> and </TROPE>.
    Following that is a story about the character "$CHARACTER$" enclosed between the tags <STORY> and </STORY>. 
    We extract the story from a movie script where the character appears.
    
    Read the story carefully and based on that answer yes or no
    if the character "$CHARACTER$" portrays or is associated with the $TROPE$ trope.
    If yes, give a brief explanation.
    If the character does not portray the trope or the story contains insufficient information, answer no.
    Answer based only on the story.
    Do not rely on your prior knowledge.
    
    <TROPE>
    $DEFINITION$
    </TROPE>

    <STORY>
    $STORY$
    </STORY>
    
    Does the character "$CHARACTER$" portray or is associated with the $TROPE$ trope in the above story?
    Answer yes or no. If yes, give a brief explanation. Do not use MarkDown."""

    system_instr = re.sub(r"[ ]*\n[ ]*", " ", system_instr)
    template = re.sub(r"\n[ ]*", "\n", template)
    print("system-instruction:======================================================================================")
    print(system_instr)
    print("=========================================================================================================\n")
    print("template:================================================================================================")
    print(template)
    print("=========================================================================================================\n")

    # creating prompts
    print("creating prompts")
    rows, prompts = [], []
    for obj in extracts:
        characterid, trope, label, name, definition, imdbid, extract = (
            obj["character"], obj["trope"], obj["label"], obj["name"], obj["definition"], obj["imdbid"], obj["text"])
        prompt = (template.replace("$TROPE$", trope).replace("$CHARACTER$", name)
                  .replace("$DEFINITION$", definition).replace("$STORY$", extract))
        rows.append([characterid, trope, imdbid, label, system_instr, template])
        prompts.append(prompt)

    # instantiate generator
    if FLAGS.llama_model is not None:
        generator = generation.Llama()
    else:
        generator = generation.Gemini(system_instr)

    # prompt
    for i in range(FLAGS.runs):
        print(f"run {i + 1}/{FLAGS.runs}")
        if FLAGS.llama_model is not None:
            responses = list(generator(prompts, system_instr))
        else:
            responses = list(generator(prompts))

        # save output
        os.makedirs(output_dir, exist_ok=True)
        output_df = pd.DataFrame(rows, columns=["character", "trope", "imdbid", "label", "system", "template"])
        output_df["response"] = responses
        output_filename = "".join(random.choice(string.ascii_letters + string.digits) for _ in range(5)) + ".csv"
        output_file = os.path.join(output_dir, output_filename)
        output_df.to_csv(output_file, index=False)

if __name__ == '__main__':
    app.run(zeroshot_extract)