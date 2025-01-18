"""List Gemini models"""
from absl import app
from absl import flags
from google import generativeai as genai

flags.DEFINE_string("key", default=None, help="Gemini key file")
FLAGS = flags.FLAGS

def list_models(_):
    gemini_key_file = FLAGS.key
    with open(gemini_key_file) as fr:
        gemini_api_key = fr.read().strip()
    genai.configure(api_key=gemini_api_key)
    for model in genai.list_models():
        print(f"{model.name:50s}   {model.display_name:50s}   input-token-limit={model.input_token_limit:7d}   "
              f"output-token-limit={model.output_token_limit:7d}")

if __name__ == '__main__':
    app.run(list_models)