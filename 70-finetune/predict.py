"""Run inference using trained LLMs on the character attribution task"""
import crm
import data
import sft

from absl import app
from absl import flags
from accelerate import PartialState
import collections

flags.DEFINE_multi_enum("dataset", default=[], enum_values=data.get_dataset_names(), help="prediction datasets")
flags.DEFINE_integer("tokenization_batch_size", default=4096, help="batch size for tokenization")
flags.DEFINE_enum("model", default="sft", enum_values=["sft", "crm"], help="prediction model")
flags.DEFINE_string("modelname", default="meta-llama/Llama-3.1-8B-Instruct", help="huggingface model name")
flags.DEFINE_string("modelpath", default=None, help="path to trained model", required=True)
flags.DEFINE_integer("prediction_batch_size", default=1, help="prediction batch size")
flags.DEFINE_bool("bf16", default=False, help="use brain floating point (default=fp16)")
flags.DEFINE_bool("load_4bit", default=False, help="load model in 4-bit")
flags.DEFINE_bool("load_8bit", default=False, help="load model in 8-bit")
flags.DEFINE_enum("attn",
                  default="sdpa",
                  enum_values=["flash_attention_2", "sdpa", "eager"],
                  help="attention implementation")

FLAGS = flags.FLAGS
PARTIALSTATE = PartialState()

def predict(_):
    """Run inference using trained LLMs on datasets for character attribution"""

    # read datasets
    datasetname_to_partitions = collections.defaultdict(set)
    datasetname_to_data = {}
    for datasetname in FLAGS.dataset:
        parts = datasetname.split("-")
        datasetname_to_partitions["-".join(parts[:-1])].add(parts[-1])
    for datasetname, partitions in datasetname_to_partitions.items():
        partitions = list(partitions)
        if datasetname.startswith("chatter-contexts"):
            _, _, preprocess, truncation_strategy, size = datasetname.split("-")
            dataset = data.load_chatter_contexts(truncation_strategy=truncation_strategy,
                                                 size_in_words=int(size),
                                                 anonymize=preprocess == "anonymized",
                                                 partitions=partitions)
        elif datasetname.startswith("chatter-segments"):
            _, _, preprocess = datasetname.split("-")
            dataset = data.load_chatter_segments(anonymize=preprocess == "anonymized", partitions=partitions)
        else:
            dataset = data.load_personet()
        for partition in partitions:
            subset = list(filter(lambda obj: obj["partition"] == partition, dataset))
            datasetname_to_data[f"{datasetname}-{partition}"] = subset

    # predict
    if FLAGS.model == "sft":
        sft.predict(partial_state=PARTIALSTATE, datasetname_to_data=datasetname_to_data)
    else:
        crm.predict(partial_state=PARTIALSTATE, datasetname_to_data=datasetname_to_data)

if __name__ == '__main__':
    app.run(predict)