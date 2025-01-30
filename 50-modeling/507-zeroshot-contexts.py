"""Zero-shot prompt extracted contexts for character attribution

Example Usage:
python 507-zeroshot-contexts.py --context_file <PATH_TO_CONTEXT_FILE> --llama_model Llama-3.1-8B-Instruct 
       --batch_size 32
"""
import datadirs
import generation

from absl import app
from absl import flags
import collections
import jsonlines
import os
import pandas as pd
import random
import re
import string

flags.DEFINE_string("context_file", default=None,
                    help="jsonlines file containing the extracted contexts, give path relative to 50-modeling/contexts")
flags.DEFINE_integer("runs", default=1, help="number of runs")
FLAGS = flags.FLAGS

def zeroshot_context(_):
    """Zero-shot prompt extracted contexts for character attribution"""
    # get file paths
    label_file = os.path.join(datadirs.datadir, "CHATTER/chatter.csv")
    tropes_file = os.path.join(datadirs.datadir, "CHATTER/tropes.csv")
    context_file = os.path.join(datadirs.datadir, "50-modeling/contexts", FLAGS.context_file)
    modelname = generation.modelname()
    context_filename = re.sub(r"\.jsonl$", "", FLAGS.context_file)
    output_dir = os.path.join(datadirs.datadir, "50-modeling/zeroshot-context", context_filename, modelname)

    # read data
    label_df = pd.read_csv(label_file, index_col=None)
    tropes_df = pd.read_csv(tropes_file, index_col=None)
    contexts = []
    trope_to_definition = {}
    with jsonlines.open(context_file) as reader:
        for obj in reader:
            contexts.append(obj)
    for _, row in tropes_df.iterrows():
        trope_to_definition[row["trope"]] = row["summary"]

    # prompt template
    system_instr = """You are a document understanding model for movie script segments.
    Give a character, and segments from a movie script where the character is mentioned or speaks, 
    and the definition of a character trope, you can accurately answer 
    whether the character portrayed or is associated with the character trope."""

    template = """A character trope is a story-telling device used by the writer to describe characters.
    Given below is the definition of the $TROPE$ trope enclosed between the tags <TROPE> and </TROPE>.
    Following that are segments from a movie script enclosed between the tags <CONTEXTS> and </CONTEXTS>. 
    The character "$CHARACTER$" is mentioned or speaks in these segments.
    
    Read the movie script segments carefully and based on that answer yes or no
    if the character "$CHARACTER$" portrays or is associated with the $TROPE$ trope.
    If yes, give a brief explanation.
    If the character does not portray the trope or the segments contains insufficient information, answer no.
    Answer based only on the movie script segments.
    Do not rely on your prior knowledge.
    
    <TROPE>
    $DEFINITION$
    </TROPE>

    <CONTEXTS>
    $CONTEXTS$
    </CONTEXTS>
    
    Does the character "$CHARACTER$" portray or is associated with the $TROPE$ trope in the above movie script segments?
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
    if "trope" in contexts[0]:
        for obj in contexts:
            characterid, trope, label, name, definition, imdbid, partition, context = (
                obj["character"], obj["trope"], obj["label"], obj["name"], obj["definition"], obj["imdbid"],
                obj["partition"], obj["text"])
            if partition == "test":
                prompt = (template.replace("$TROPE$", trope).replace("$CHARACTER$", name)
                          .replace("$DEFINITION$", definition).replace("$CONTEXTS$", context))
                rows.append([characterid, trope, imdbid, label, system_instr, template])
                prompts.append(prompt)
    else:
        characterid_to_ixs = collections.defaultdict(list)
        for i, obj in enumerate(contexts):
            characterid_to_ixs[obj["character"]].append(i)
        for _, row in label_df[label_df["partition"] == "test"].iterrows():
            characterid, trope, label = row["character"], row["trope"], row["label"]
            definition = trope_to_definition[trope]
            for i in characterid_to_ixs[characterid]:
                imdbid, name, context = contexts[i]["imdbid"], contexts[i]["name"], contexts[i]["text"]
                prompt = (template.replace("$TROPE$", trope).replace("$CHARACTER$", name)
                          .replace("$DEFINITION$", definition).replace("$CONTEXTS$", context))
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
    app.run(zeroshot_context)