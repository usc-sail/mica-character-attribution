"""Zero-shot prompt the movie script to find whether character portrays the trope

Example Usage:
python 52-zeroshot-script.py --llama_model Llama-3.1-8B-Instruct --batch_size 1 --max_input_tokens 64 --max_output_tokens 256 --temperature 1
python 52-zeroshot-script.py --gemini_model gemini-1.5-flash --gemini_key <PATH_TO_GEMINI_KEY_FILE> --max_output_tokens 256 --temperature 1
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
import tqdm

flags.DEFINE_integer("runs", default=1, help="number of runs")
flags.DEFINE_bool("evaluate", default=False, help="evaluate the response")
FLAGS = flags.FLAGS

def evaluate_response():
    """Evaluate the generator's response"""
    output_dir = os.path.join(datadirs.datadir, f"50-prompting/zeroshot-script-{FLAGS.model}")
    acc_arr, prec_arr, rec_arr, f1_arr = [], [], [], []
    for file in os.listdir(output_dir):
        if file.endswith(".csv"):
            output_file = os.path.join(output_dir, file)
            output_df = pd.read_csv(output_file, index_col=None)
            output_df["answer"] = output_df["response"].str.extract(r"(\w+)")[0].str.lower()
            output_df["pred"] = output_df["answer"].apply(
                lambda ans: 1 if ans == "yes" else 0 if ans == "no" else np.nan)
            true = []
            pred = []
            for _, df in output_df.groupby(["character", "trope"]):
                preds = df["pred"].dropna().values
                if len(preds) > 0:
                    pred.append(int(np.any(preds == 1).item()))
                else:
                    pred.append(np.nan)
                true.append(df["label"].values[0])
            true = np.array(true)
            pred = np.array(pred)
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

def prompt(_):
    """Zero-shot prompt the scripts to find character attribution"""
    # get file paths
    movie_scripts_dir = os.path.join(datadirs.datadir, "movie-scripts")
    label_file = os.path.join(datadirs.datadir, "CHATTER/chatter.csv")
    map_file = os.path.join(datadirs.datadir, "CHATTER/character-movie-map.csv")
    tropes_file = os.path.join(datadirs.datadir, "CHATTER/tropes.csv")
    output_dir = os.path.join(datadirs.datadir, f"50-prompting/zeroshot-script/{FLAGS.model}")

    # get the character name and the scripts where they appear
    print("read data")
    label_df = pd.read_csv(label_file, index_col=None)
    map_df = pd.read_csv(map_file, index_col=None, dtype=str)
    tropes_df = pd.read_csv(tropes_file, index_col=None)
    imdbid_to_script = {}
    characterid_to_name = {}
    characterid_to_imdbids = {}
    trope_to_definition = {}
    for imdbid in tqdm.tqdm(map_df["imdb-id"].unique(), desc="read movie scripts"):
        script_file = os.path.join(movie_scripts_dir, imdbid, "script.txt")
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
    system_content = """You are a document understanding model for movie scripts.
    Given a movie script, a character who appears in the movie script, and the definition of a character trope, 
    you can accurately answer whether the character portrayed or is associated with the character trope."""
    template = """Given below is a movie script enclosed between the tags <SCRIPT> and </SCRIPT>.
    The character '$CHARACTER$' appears in the movie script.
    Following that is the definition of the $TROPE$ trope enclosed between the tags <TROPE> and </TROPE>.
    
    Read the movie script carefully and based on that answer yes or no
    if the character "$CHARACTER$" portrays or is associated with the $TROPE$ " trope.
    If yes, give a brief explanation.
    Answer based only on the movie script.
    Do not rely on your " prior knowledge.
    
    <SCRIPT>
    $SCRIPT$
    </SCRIPT>
    
    <TROPE>
    $DEFINITION$
    </TROPE>
    
    Does the character "$CHARACTER$" portray or is associated with the $TROPE$ trope in the above movie script?
    Answer yes or no. If yes, give a brief explanation. Do not use MarkDown."""
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
    print("creating prompts")
    rows, prompts = [], []
    test_df = label_df[label_df["partition"] == "test"]
    os.makedirs(output_dir, exist_ok=True)
    for _, row in test_df.iterrows():
        characterid, trope, label = row.character, row.trope, row.label
        name = characterid_to_name[characterid]
        definition = trope_to_definition[trope]
        for imdbid in characterid_to_imdbids[characterid]:
            script = imdbid_to_script[imdbid]
            prompt = (template.replace("$TROPE$", trope).replace("$CHARACTER$", name)
                      .replace("$DEFINITION$", definition).replace("$SCRIPT$", script))
            rows.append([characterid, trope, label, imdbid])
            prompts.append(prompt)

    # prompt
    for i in range(1, FLAGS.runs + 1):
        if FLAGS["gemini-model"].value is not None:
            responses = generator(prompts)
        else:
            responses = generator(prompts, system_instr, padding="max_length", truncation=True)
        output_file = os.path.join(output_dir, f"run{i}.csv")
        output_df = pd.DataFrame(rows, columns=["character", "trope", "label", "imdbid"])
        output_df["response"] = responses
        output_df.to_csv(output_file, index=False)

def main(_):
    if FLAGS.evaluate:
        evaluate_response()
    else:
        prompt()

if __name__ == '__main__':
    app.run(main)