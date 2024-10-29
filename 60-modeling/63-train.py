"""Train the character representation models. This file is the main function that collects the hyperparameters
and input/output files, trains the model, and evaluates the trained model
"""
from absl import app
from absl import flags

from trainers import trainer

FLAGS = flags.FLAGS

# input-output
flags.DEFINE_string("datadir", default=None, help="data directory", required=True)
flags.DEFINE_enum("inputtype", default="story", enum_values=["story", "character"],
                  help="encode complete story or character segments")
flags.DEFINE_string("tokenizermodel", default=None,
                    help=("tokenizer used to preprocess the data; should be a huggingface model; if not provided, it "
                          "defaults to the pretrained_model_name"))

# model params
flags.DEFINE_string("pretrainedmodel", default="roberta-base",
                    help="base encoder to use; should be a huggingface model")
flags.DEFINE_bool("labeldependent", default=False, help="set to use label-dependent model")

# training params
flags.DEFINE_integer("nepochs", default=10, help="number of epochs")
flags.DEFINE_float("encoderlr", default=2e-5, help="learning rate of the encoder")
flags.DEFINE_float("lr", default=1e-3, help="learning rate of modules apart from the encoder")
flags.DEFINE_bool("freezeencoder", default=False, help="set to freeze encoder")

# batching params
flags.DEFINE_integer("ncharactersbatch", default=5,
                     help="number of characters per batch when training with data type = 'character'")
flags.DEFINE_integer("ntropesbatch", default=50,
                     help="number of tropes per batch when training with data type = 'story'")

def start_training(_):
    print("\n")
    FLAGS.tokenizermodel = FLAGS.tokenizermodel if FLAGS.tokenizermodel else FLAGS.pretrainedmodel
    Trainer = trainer.Trainer(FLAGS)
    Trainer()

if __name__ == '__main__':
    app.run(start_training)