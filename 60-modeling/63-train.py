"""Train the character representation models. This file is the main function that collects the hyperparameters
and input/output files, trains the model, and evaluates the trained model
"""
from trainers import trainer

from absl import app
from absl import flags

FLAGS = flags.FLAGS

# input-output
flags.DEFINE_string("datadir", default=None, help="data directory", required=True)
flags.DEFINE_enum("inputtype", default="story", enum_values=["story", "character"],
                  help="encode complete story or character segments")
flags.DEFINE_string("tokenizermodel", default=None,
                    help=("tokenizer used to preprocess the data; should be a huggingface model; if not provided, it "
                          "defaults to the pretrained_model_name"))
flags.DEFINE_string("datasetfile", default="60-modeling/dataset-with-only-character-tropes.csv",
                    help=("csv file containing the character, movie, and trope data; columns include imdb-id, "
                          "character, trope, component, partition, label"))
flags.DEFINE_string("tropesfile", default="60-modeling/character-tropes.csv",
                    help="csv file containing trope name and definition")

# model params
flags.DEFINE_string("pretrainedmodel", default="roberta-base",
                    help="base encoder to use; should be a huggingface model")
flags.DEFINE_bool("labeldependent", default=False, help="set to use label-dependent model")
flags.DEFINE_integer("nepochs", default=10, help="number of epochs")
flags.DEFINE_float("encoderlr", default=2e-5, help="learning rate of the encoder")
flags.DEFINE_float("lr", default=1e-3, help="learning rate of modules apart from the encoder")
flags.DEFINE_bool("freezeencoder", default=False, help="set to freeze encoder")
flags.DEFINE_integer("ncharactersbatch", default=5, help=("number of characters per batch when training with "
                                                          "data type = character"))

def start_training(_):
    tokenizer_model = FLAGS.tokenizermodel if FLAGS.tokenizermodel else FLAGS.pretrainedmodel
    Trainer = trainer.Trainer(FLAGS.datadir,
                              FLAGS.datasetfile,
                              FLAGS.tropesfile,
                              FLAGS.inputtype,
                              FLAGS.pretrainedmodel,
                              tokenizer_model,
                              FLAGS.labeldependent,
                              FLAGS.nepochs,
                              FLAGS.encoderlr,
                              FLAGS.lr,
                              FLAGS.freezeencoder,
                              FLAGS.ncharactersbatch
                              )
    Trainer()

if __name__ == '__main__':
    app.run(start_training)