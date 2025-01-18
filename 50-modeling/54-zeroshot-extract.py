"""Zero-shot prompt extracted sections for character attribution"""
import datadirs

from absl import app
from absl import flags
import json
import os

flags.DEFINE_string("extract-file", default=None,
                    help="json file containing the extracted sections, give path relative to 60-modeling/extracts")
flags.DEFINE_string("model", default="Llama-3.1-8B-Instruct", help="Llama model")
flags.DEFINE_string("batch-size", default=1, help="batch size")
flags.DEFINE_string("evaluate", default=False, help="evaluate the prompt responses")
FLAGS = flags.FLAGS

def zeroshot_sections(_):
    """Zero-shot prompt extracted sections for character attribution"""
    # read data
    extract_file = os.path.join(datadirs.datadir, "60-modeling/extracts", FLAGS["extract-file"].value)
    extract_data = json.load(open(extract_file))

    # 