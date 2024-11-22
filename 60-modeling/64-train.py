"""Train the character representation models. This file is the main function that collects the hyperparameters
and input/output files, trains the model, and evaluates the trained model
pass --alsologtostderr to log to both log file and stderr
pass --logtostderr to only log to stderr
"""
import os
from absl import app
from absl import flags
from absl import logging
import datetime

import datadirs
from trainers import trainer

FLAGS = flags.FLAGS

# input-output
flags.DEFINE_enum("input", default="story", enum_values=["story", "character"], help="input")

# model params
flags.DEFINE_enum("model", default="roberta", enum_values=["roberta", "longformer"], help="encoder model")
flags.DEFINE_integer("longformerattn", default=128, help="sliding attention window of longformer model")
flags.DEFINE_bool("lbl", default=True, help="use label-dependent model")

# peft params
flags.DEFINE_float("alpha", default=32, help="lora scaling factor")
flags.DEFINE_integer("rank", default=8, help="lora rank")
flags.DEFINE_float("dropout", default=0.1, help="lora dropout")

# training params
flags.DEFINE_integer("ep", default=1, help="epochs")
flags.DEFINE_float("elr", default=2e-5, help="learning rate of the encoder")
flags.DEFINE_float("lr", default=1e-3, help="learning rate of modules apart from the encoder")
flags.DEFINE_float("gradnorm", default=10, help="maximum gradient norm")
flags.DEFINE_float("decay", default=1e-2, help="weight decay")

# batching params
flags.DEFINE_integer("chrbatch", default=5, help="characters per batch when training with input = 'character'")
flags.DEFINE_integer("trpbatch", default=50, help="tropes per batch when training with input = 'story'")
flags.DEFINE_integer("tokbatch", default=40, help="tokens in '000s per batch")
flags.DEFINE_integer("batchepoch", default=100, help="number of batches per epoch")
flags.DEFINE_integer("batcheval", default=25, help="number of dev batches to evaluate per epoch")

def start_training(_):
    now = datetime.datetime.now()
    nowtext = datetime.datetime.strftime(now, "%Hh%Mm%Ss-%Y%b%d")
    expdir = os.path.join(datadirs.datadir, "60-modeling/experiments", f"{nowtext}-{FLAGS.input}-{FLAGS.model}")
    os.makedirs(expdir, exist_ok=True)
    logging.get_absl_handler().use_absl_log_file(program_name="train", log_dir=expdir)
    logging.get_absl_handler().setFormatter(None)

    logging.info(f"{'experiments dir':80s} = {expdir}")
    for flagobj in FLAGS.flags_by_module_dict()["60-modeling/64-train.py"]:
        logging.info(f"{flagobj.name:20s} {flagobj.help:70s} = {flagobj.value}")
    logging.info("")

    Trainer = trainer.Trainer(FLAGS, expdir)
    Trainer()
    try:
        os.rmdir(expdir)
        logging.info("experiments dir removed")
    except OSError:
        logging.info("experiments dir not removed because it is not empty")

if __name__ == '__main__':
    app.run(start_training)