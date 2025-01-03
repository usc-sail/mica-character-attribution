"""Prompt Llama on the test dataset for character attribution using movie scripts"""
import pandas as pd
import tqdm
import os

import transformers
import torch

from absl import app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_string("datadir", default=None, help="data directory", required=True)
flags.DEFINE_string("datafile", default="60-modeling/dataset-with-only-character-tropes.csv", help="dataset file")
flags.DEFINE_string("testdatafile", default="60-modeling/test-dataset.csv", help="test dataset file")
flags.DEFINE_string("tropesfile", default="60-modeling/tropes.csv", help="tropes definitions file")
flags.DEFINE_string("prompttemplatefile", default="50-prompting/prompt-templates/llama-script-attribution.txt",
                    help="text file containing the prompt template")
flags.DEFINE_string("moviescriptsdir", default="movie-scripts", help="movie scripts dir")
flags.DEFINE_string("outputresponsefile", default="50-prompting/prompt-responses/llama-script-attribution.csv",
                    help="character attribution response file")
flags.DEFINE_integer("candidates", default=5, help="number of candidate generations")
flags.DEFINE_string("llamamodel", default="meta-llama/Meta-Llama-3.1-8B-Instruct", help="Llama model")
flags.DEFINE_float("temperature", default=1, help="temperature of generation")

def prompt_for_character_attribution(_):
    data_dir = FLAGS.datadir
    data_file = os.path.join(data_dir, FLAGS.datafile)
    test_file = os.path.join(data_dir, FLAGS.testdatafile)
    tropes_file = os.path.join(data_dir, FLAGS.tropesfile)
    template_file = os.path.join(data_dir, FLAGS.prompttemplatefile)
    movie_scripts_dir = os.path.join(data_dir, FLAGS.moviescriptsdir)
    output_response_file = os.path.join(data_dir, FLAGS.outputresponsefile)
    ncandidates = FLAGS.candidates
    model_name = FLAGS.llamamodel
    temperature = FLAGS.temperature

    data_df = pd.read_csv(data_file, index_col=None, dtype={"imdb-id": str, "content-text": str})
    test_df = pd.read_csv(test_file, index_col=None, dtype={"combination": str})
    tropes_df = pd.read_csv(tropes_file, index_col=None)

    test_imdbids = sorted(set([imdbid for imdbids in test_df["imdb-ids"] for imdbid in imdbids.split(";")]))
    imdbid_to_script = {}
    for imdbid in test_imdbids:
        script_file = os.path.join(movie_scripts_dir, imdbid, "script.txt")
        with open(script_file) as fr:
            imdbid_to_script[imdbid] = fr.read().strip()

    character_and_imdbid_to_name = {}
    for character, imdbid, name in (data_df[["character", "imdb-id", "imdb-character"]]
                                    .drop_duplicates().itertuples(index=False, name=None)):
        character_and_imdbid_to_name[(character, imdbid)] = name

    trope_to_definition = {}
    for trope, definition in tropes_df.itertuples(index=False, name=None):
        trope_to_definition[trope] = definition

    with open(template_file) as fr:
        template = fr.read().strip()

    prompt_rows = []
    for character, trope, imdbids in test_df[["character", "trope", "imdb-ids"]].itertuples(index=False, name=None):
        for imdbid in imdbids.split(";"):
            script = imdbid_to_script[imdbid]
            definition = trope_to_definition[trope]
            name = character_and_imdbid_to_name[(character, imdbid)]
            prompt = (template
                      .replace("$$CHARACTER$$", name)
                      .replace("$$TROPE$$", trope)
                      .replace("$$DEFINITION$$", definition)
                      .replace("$$SCRIPT$$", script))
            prompt_rows.append([character, trope, imdbid, prompt])
    prompt_df = pd.DataFrame(prompt_rows, columns=["character", "trope", "imdb-id", "prompt"])

    pipeline = transformers.pipeline(model=model_name,
                                     task="text-generation",
                                     model_kwargs={"torch_dtype": torch.bfloat16,
                                                   "cache_dir": "/project/shrikann_35/llm-shared"},
                                     device_map="auto")

    responses_arr = []
    for prompt in tqdm.tqdm(prompt_df["prompt"], desc="prompting", unit="prompt"):
        output = pipeline(prompt, max_new_tokens=5, return_full_text=False, num_return_sequences=ncandidates,
                          temperature=temperature)
        responses = [output[i]["generated_text"] for i in range(ncandidates)]
        responses_arr.append(responses)

    response_df = pd.DataFrame(responses_arr, columns=[f"response-{i + 1}" for i in range(ncandidates)])
    response_df = pd.concat([prompt_df[["character", "trope", "imdb-id"]], response_df], axis=1)
    response_df.to_csv(output_response_file, index=False)

if __name__ == '__main__':
    app.run(prompt_for_character_attribution)