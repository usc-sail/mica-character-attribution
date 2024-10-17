"""Prompt Gemini on the test dataset for character attribution using movie scripts"""
import pandas as pd
import time
import tqdm
import os

from google.api_core import retry
from google import generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from vertexai.preview import tokenization

from absl import app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_string("datadir", default=None, help="data directory", required=True)
flags.DEFINE_string("geminikeyfile", default=None, help="file containing Gemini key", required=True)
flags.DEFINE_bool("r", default=False, help="set to prompt, otherwise only the price is estimated")
flags.DEFINE_string("datafile", default="60-modeling/dataset-with-only-character-tropes.csv", help="dataset file")
flags.DEFINE_string("testdatafile", default="60-modeling/test-dataset.csv", help="test dataset file")
flags.DEFINE_string("tropesfile", default="60-modeling/tropes.csv", help="tropes definitions file")
flags.DEFINE_string("prompttemplatefile", default="50-prompting/prompt-templates/gemini-script-attribution.txt",
                    help="text file containing the prompt template")
flags.DEFINE_string("moviescriptsdir", default="movie-scripts", help="movie scripts dir")
flags.DEFINE_string("outputresponsefile", default="50-prompting/prompt-responses/gemini-script-attribution.csv",
                    help="character attribution response file")
flags.DEFINE_integer("candidates", default=5, help="number of candidate generations")
flags.DEFINE_string("geminimodel", default="gemini-1.5-flash-002", help="Gemini model")
flags.DEFINE_float("temperature", default=1, help="temperature of generation")
flags.DEFINE_float("tokenrate", default=0.075, help="input token rate per million tokens")

def prompt_for_character_attribution(_):
    data_dir = FLAGS.datadir
    gemini_key_file = FLAGS.geminikeyfile
    run = FLAGS.r
    data_file = os.path.join(data_dir, FLAGS.datafile)
    test_file = os.path.join(data_dir, FLAGS.testdatafile)
    tropes_file = os.path.join(data_dir, FLAGS.tropesfile)
    template_file = os.path.join(data_dir, FLAGS.prompttemplatefile)
    movie_scripts_dir = os.path.join(data_dir, FLAGS.moviescriptsdir)
    output_response_file = os.path.join(data_dir, FLAGS.outputresponsefile)
    ncandidates = FLAGS.candidates
    model_name = FLAGS.geminimodel
    temperature = FLAGS.temperature
    trr = FLAGS.tokenrate

    with open(gemini_key_file) as fr:
        gemini_api_key = fr.read().strip()
    genai.configure(api_key=gemini_api_key)

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

    tokenizer = tokenizer = tokenization.get_tokenizer_for_model(model_name)

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

    print("calculating prompt sizes")
    prompt_df["prompt-size"] = prompt_df["prompt"].apply(lambda prompt: tokenizer.count_tokens(prompt).total_tokens)
    estimated_cost = (trr / 1e6) * prompt_df["prompt-size"].sum()
    print(f"estimated total cost = {estimated_cost:.2f}\n")

    if run:
        model = genai.GenerativeModel(model_name=model_name)
        safety_settings={HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE}

        responses_arr = []
        for prompt in tqdm.tqdm(prompt_df["prompt"], desc="prompting", unit="prompt"):
            output = model.generate_content(prompt,
                                            safety_settings=safety_settings,
                                            generation_config=genai.types.GenerationConfig(
                                                candidate_count=ncandidates,
                                                temperature=temperature),
                                            request_options=genai.types.RequestOptions(
                                                retry=retry.Retry(initial=10, multiplier=2, maximum=60, timeout=300)))
            try:
                responses = [candidate.content.parts[0].text for candidate in output.candidates]
            except Exception:
                responses = [(f"ERROR: prompt-feedback = {output.prompt_feedback} "
                            f"finish-reason = {candidate.finish_reason} "
                            f"safety-ratings = {candidate.safety_ratings}")
                            for candidate in output.candidates]
            responses_arr.append(responses)

        response_df = pd.DataFrame(responses_arr, columns=[f"response-{i + 1}" for i in range(ncandidates)])
        response_df = pd.concat([prompt_df[["character", "trope", "imdb-id"]], response_df], axis=1)
        response_df.to_csv(output_response_file, index=False)

if __name__ == '__main__':
    app.run(prompt_for_character_attribution)