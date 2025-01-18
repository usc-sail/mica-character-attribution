"""Prompt to find the prior knowledge of the model

Example Usage:
python 51-priors.py --llama_model Llama-3.1-8B-Instruct --batch_size 1 --max_output_tokens 256 --temperature 1
python 51-priors.py --gemini_model gemini-1.5-flash --gemini_key <PATH_TO_GEMINI_KEY_FILE> --max_output_tokens 256 --temperature 1
"""
import datadirs
import generation

from absl import app
from absl import flags
import numpy as np
import os
import pandas as pd
import re
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

flags.DEFINE_integer("runs", default=5, help="number of runs")
flags.DEFINE_bool("evaluate", default=False, help="evaluate the response")
FLAGS = flags.FLAGS

def evaluate_response():
    """Evaluate the prior knowledge of the models"""
    output_dir = os.path.join(datadirs.datadir, f"50-modeling/priors/{FLAGS.model}")
    acc_arr, prec_arr, rec_arr, f1_arr = [], [], [], []
    for file in os.listdir(output_dir):
        if file.endswith(".csv"):
            output_file = os.path.join(output_dir, file)
            output_df = pd.read_csv(output_file, index_col=None)
            output_df["answer"] = output_df["response"].str.extract(r"(\w+)")[0].str.lower()
            true = output_df["label"].astype(int)
            pred = output_df["answer"].apply(lambda ans: 1 if ans == "yes" else 0 if ans == "no" else np.nan).values
            n = np.isnan(pred).sum()
            true = true[~np.isnan(pred)]
            pred = pred[~np.isnan(pred)]
            acc = accuracy_score(true, pred)
            prec, rec, f1, _ = precision_recall_fscore_support(true, pred, average="binary")
            print(f"{file}")
            print(f"could not parse {n} samples")
            print(f"n={len(true)} samples evaluated")
            print(f"acc={acc:.3f} precision={prec:.3f} recall={rec:.3f} F1={f1:.3f}\n")
            acc_arr.append(acc)
            prec_arr.append(prec)
            rec_arr.append(rec)
            f1_arr.append(f1)
    print(f"acc: mean={np.mean(acc_arr):.3f} std={np.std(acc_arr):.4f}")
    print(f"precision: mean={np.mean(prec_arr):.3f} std={np.std(prec_arr):.4f}")
    print(f"recall: mean={np.mean(rec_arr):.3f} std={np.std(rec_arr):.4f}")
    print(f"f1: mean={np.mean(f1_arr):.3f} std={np.std(f1_arr):.4f}")

def prompt_priors(_):
    """Prompt models to find prior knowledge"""
    # read data
    print("read data")
    data_df = pd.read_csv(os.path.join(datadirs.datadir, "CHATTER/chatter.csv"), index_col=None)
    map_df = pd.read_csv(os.path.join(datadirs.datadir, "CHATTER/character-movie-map.csv"), index_col=None, dtype=str)
    metadata_df = pd.read_csv(os.path.join(datadirs.datadir, "CHATTER/movie-metadata.csv"), index_col=None, dtype=str)
    tropes_df = pd.read_csv(os.path.join(datadirs.datadir, "CHATTER/tropes.csv"), index_col=None)
    output_dir = os.path.join(datadirs.datadir, f"50-prompting/priors/{FLAGS.model}")

    # process data
    print("process data")
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
    print("system-instruction:======================================================================================")
    print(system_instr)
    print("=========================================================================================================\n")
    print("template:================================================================================================")
    print(template)
    print("=========================================================================================================\n")

    # instantiate generator
    print("instantiating generator")
    if FLAGS["gemini-model"].value is not None:
        generator = generation.Gemini(system_instr)
    else:
        generator = generation.Llama()

    # create prompts
    print("create prompts")
    os.makedirs(output_dir, exist_ok=True)
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
    for i in range(1, FLAGS.runs + 1):
        if FLAGS["gemini-model"].value is not None:
            responses = generator(prompts)
        else:
            responses = generator(prompts, system_instr)
        output_file = os.path.join(output_dir, f"run{i}.csv")
        output_df = pd.DataFrame(rows, columns=["character", "trope", "label", "system-instr", "prompt"])
        output_df["response"] = responses
        output_df.to_csv(output_file, index=False)

def main(_):
    if FLAGS.evaluate:
        evaluate_response()
    else:
        prompt_priors()

if __name__ == '__main__':
    app.run(main)