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
import shutil

import datadirs
from trainers import trainer

FLAGS = flags.FLAGS

# input-output
flags.DEFINE_bool("segment", default=False, help="set to use character segment input")
flags.DEFINE_enum("context", default="0", enum_values=["0", "5", "10"],
                  help="number of neighboring segments for character input")
flags.DEFINE_bool("save_logs", default=True, help="save log file")

# model params
flags.DEFINE_enum("model", default="roberta", enum_values=["roberta", "longformer", "llama"], help="encoder model")
flags.DEFINE_integer("longformerattn", default=128, help="sliding attention window of longformer model")
flags.DEFINE_bool("lbl", default=True, help="use label-dependent model")
flags.DEFINE_bool("mention", default=True, help="use character mentions")
flags.DEFINE_bool("utter", default=True, help="use character utterances")

# peft params
flags.DEFINE_bool("lora", default=True, help="use LoRA")
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
flags.DEFINE_integer("trpbatch", default=50, help="tropes per batch")
flags.DEFINE_integer("tokbatch", default=50, help="tokens in '000s per batch")
flags.DEFINE_integer("batchepoch", default=100, help="number of batches per epoch")
flags.DEFINE_integer("batcheval", default=None, help="number of dev batches to evaluate per epoch")

def start_training(_):
    now = datetime.datetime.now()
    nowtext = datetime.datetime.strftime(now, "%b%d-%H:%M:%S")
    inputtype = "character" if FLAGS.segment else "story"
    modeltype = "labeldep" if FLAGS.lbl else "labelind"
    expdir = os.path.join(datadirs.datadir, "60-modeling/experiments",
                          f"{nowtext}-{modeltype}-{inputtype}-{FLAGS.model}")
    os.makedirs(expdir, exist_ok=True)
    logging.get_absl_handler().use_absl_log_file(program_name="train", log_dir=expdir)
    logging.get_absl_handler().setFormatter(None)

    logging.info(f"{'experiments dir':80s} = {expdir}")
    for module in FLAGS.flags_by_module_dict():
        if module.endswith("64-train.py"):
            for flagobj in FLAGS.flags_by_module_dict()[module]:
                logging.info(f"{flagobj.name:20s} {flagobj.help:70s} = {flagobj.value}")
    logging.info("")

    Trainer = trainer.Trainer(FLAGS, expdir)
    Trainer()

    if not FLAGS.save_logs:
        logging.info("experiments dir removed")
        shutil.rmtree(expdir)

if __name__ == '__main__':
    app.run(start_training)