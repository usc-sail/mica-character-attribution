"""Train the character representation models. This file is the main function that collects the hyperparameters
and input/output files, trains the model, and evaluates the trained model
"""
from trainers import trainer

from absl import app
from absl import flags

FLAGS = flags.FLAGS

# input-output
flags.DEFINE_string("data_dir", default=None, help="data directory", required=True)
flags.DEFINE_enum("data_type", default="story", enum_values=["story", "character"],
                  help="encode complete story or character segments")
flags.DEFINE_string("tokenizer_model_name", default=None,
                    help=("tokenizer used to preprocess the data; should be a huggingface model; if not provided, it "
                          "defaults to the pretrained_model_name"))
flags.DEFINE_string("splits_file", default="20-preprocessing/train-dev-test-splits-with-negatives.csv",
                    help=("csv file containing the train-dev-test splits; columns include imdb-id, character, trope, "
                          "component, partition, label"))
flags.DEFINE_string("tropes_file", default="20-preprocessing/tropes-with-antonyms.csv",
                    help="csv file containing trope name and definition")

# model params
flags.DEFINE_string("pretrained_model_name", default="roberta-base",
                    help="base encoder to use; should be a huggingface model")
flags.DEFINE_bool("label_dependent", default=False, help="set to use label-dependent model")

def start_training(_):
    tokenizer_model_name = FLAGS.tokenizer_model_name if FLAGS.tokenizer_model_name else FLAGS.pretrained_model_name
    Trainer = trainer.Trainer(FLAGS.data_dir,
                              FLAGS.splits_file,
                              FLAGS.tropes_file,
                              FLAGS.data_type,
                              FLAGS.pretrained_model_name,
                              tokenizer_model_name,
                              FLAGS.label_dependent,
                              )

if __name__ == '__main__':
    app.run(start_training)