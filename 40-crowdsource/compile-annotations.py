"""Compile the annotations from the MTurk results files"""
import os
import krippendorff
import numpy as np
import pandas as pd
import itertools

from absl import app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_string("data_dir", default=None, help="data directory", required=True)
flags.DEFINE_string("batches", default="40-crowdsource/mturk-batch-results/batchfiles.txt",
                    help="text file containing the file paths to the MTurk annotation results of the batches")
flags.DEFINE_string("unreliableworkerids", default="40-crowdsource/mturk-workers-performance/unreliable-worker-ids.txt",
                    help="text file containing the worker ids of the unreliable workers")
flags.DEFINE_string("datasettoannotate", default="40-crowdsource/dataset-to-annotate.csv",
                    help="csv file of the dataset to annotate")
flags.DEFINE_string("outputannotation", default="40-crowdsource/mturk-aggregate-results/annotations.csv",
                    help="csv file containing the per worker annotation of the samples")
flags.DEFINE_string("outputaggregate", default="40-crowdsource/mturk-aggregate-results/aggregate.csv",
                    help="csv file containing the aggregate annotation of the samples")
flags.DEFINE_string("outputcombination", default="40-crowdsource/mturk-aggregate-results/combinations.csv",
                    help="csv file containing distribution of label combinations")

def annotation_to_annotation_ambiguity(annotation):
    if annotation in ["no", "yes"]:
        return 0
    elif annotation in ["maybeno", "maybeyes"]:
        return 1
    else:
        return 2

def compile_annotations(_):
    data_dir = FLAGS.data_dir
    batches_file = os.path.join(data_dir, FLAGS.batches)
    unreliable_workerids_file = os.path.join(data_dir, FLAGS.unreliableworkerids)
    dataset_file = os.path.join(data_dir, FLAGS.datasettoannotate)
    aggregate_file = os.path.join(data_dir, FLAGS.outputaggregate)
    annotation_file = os.path.join(data_dir, FLAGS.outputannotation)
    combination_file = os.path.join(data_dir, FLAGS.outputcombination)

    # read mturk batch responses
    with open(batches_file) as fr:
        annotation_files = fr.read().strip().split("\n")
    annotation_files = [os.path.join(data_dir, annotation_file) for annotation_file in annotation_files]
    print(f"concatenate {len(annotation_files)} annotation files")

    # concatenate mturk batch responses
    annotation_dfs = []
    for fp in annotation_files:
        annotation_df = pd.read_csv(fp, index_col=None)
        annotation_dfs.append(annotation_df)
    annotation_df = pd.concat(annotation_dfs, axis=0)
    n = len(annotation_df)
    print(f"{len(annotation_df)} annotations\n")

    # retain only one annotation from a worker per (character, trope) sample
    print("retain only one annotation from a worker per (character, trope) sample")
    annotation_df = annotation_df[["Input.character", "Input.trope", "WorkerId", "AssignmentId", "WorkTimeInSeconds",
                                   "Answer.clickedFandom", "Answer.clickedTVTrope", "Answer.clickedWikipedia",
                                   "Answer.comments", "Answer.portray"]]
    annotation_df.columns = ["character", "trope", "worker-id", "assignment-id", "worktime-in-seconds",
                             "annotation-clicked-fandom", "annotation-clicked-tvtrope",
                             "annotation-clicked-wikipedia", "annotation-comments", "annotation"]
    annotation_df["annotation-ambiguity"] = annotation_df["annotation"].apply(annotation_to_annotation_ambiguity)
    annotation_df.sort_values(by=["character", "trope", "worker-id", "annotation-ambiguity"], inplace=True)
    annotation_df.drop_duplicates(subset=("character", "trope", "worker-id"), inplace=True)
    annotation_df.drop(columns=["annotation-ambiguity"], inplace=True)
    print(f"{n - len(annotation_df)} annotations removed")
    print(f"{len(annotation_df)} annotations\n")

    # read unreliable worker ids
    with open(unreliable_workerids_file) as fr:
        unreliable_workerids = set(fr.read().strip().split("\n"))
    print(f"{len(unreliable_workerids)} unreliable worker ids")

    # add a column to indicate worker reliability
    annotation_df["reliable"] = True
    annotation_df.loc[annotation_df["worker-id"].isin(unreliable_workerids), "reliable"] = False
    n = len(annotation_df)
    n_unreliable_annotations = (~annotation_df["reliable"]).sum()
    percent_unreliable_annotations = 100 * n_unreliable_annotations / n
    print(f"{n_unreliable_annotations} unreliable annotations ({percent_unreliable_annotations:.1f}%)")
    print(f"{n - n_unreliable_annotations} reliable annotations ({100 - percent_unreliable_annotations:.1f}%)")

    # merge dataset with annotation to find how many samples have been partially annotated
    dataset_df = pd.read_csv(dataset_file, index_col=None, dtype={"imdb-id": str, "content-text": str})
    dataset_df = dataset_df.merge(annotation_df, how="left", on=["character", "trope"])
    dataset_df.dropna(subset="annotation", inplace=True)

    # calculate krippendorff's score
    workerids = dataset_df["worker-id"].unique()
    unitids = list(dataset_df[["character", "trope"]].drop_duplicates().itertuples(index=False, name=None))
    reliability_df = pd.DataFrame(index=workerids, columns=unitids)
    for character, trope, workerid, annotation in (dataset_df[["character", "trope", "worker-id", "annotation"]]
                                                   .itertuples(index=False, name=None)):
        reliability_df.loc[workerid, (character, trope)] = annotation
    krippendorff.alpha()

    # create aggregate annotations file
    aggregate_rows = []
    for (character, trope), sample_df in dataset_df.groupby(["character", "trope"]):
        annotations = sample_df["annotation"].values
        reliable = sample_df["reliable"].values

        # find reliable annotations
        n_unreliable = (reliable == False).sum()
        annotations = annotations[reliable == True]

        # find definite annotations
        sure = (annotations != "notsure")
        n_notsure = (sure == False).sum()
        annotations = annotations[sure == True]
        n = len(annotations)

        # check that number of annotations is at most 3
        assert len(annotations) <= 3

        # find number of yes, maybeyes, maybeno, and no values
        n_yes = (annotations == "yes").sum()
        n_maybeyes = (annotations == "maybeyes").sum()
        n_maybeno = (annotations == "maybeno").sum()
        n_no = (annotations == "no").sum()

        # create aggregate code label
        label = f"{n_yes}{n_maybeyes}{n_maybeno}{n_no}"

        # create aggregate row
        aggregate_row = [character, trope, n_unreliable, n_notsure, n, n_yes, n_maybeyes, n_maybeno, n_no, label]
        aggregate_rows.append(aggregate_row)

    aggregate_df = pd.DataFrame(aggregate_rows, columns=["character", "trope", "unreliable", "notsure", "n",
                                                         "yes", "maybeyes", "maybeno", "no", "combination"])

    # create combinations distribution file
    combination_rows = []
    labels = ["yes", "maybeyes", "maybeno", "no"]
    for r in range(4):
        for combination in itertools.combinations_with_replacement(labels, r):
            combination_arr = np.array(combination)
            combination_n_yes = (combination_arr == "yes").sum()
            combination_n_maybeyes = (combination_arr == "maybeyes").sum()
            combination_n_maybeno = (combination_arr == "maybeno").sum()
            combination_n_no = (combination_arr == "no").sum()
            combination_label = f"{combination_n_yes}{combination_n_maybeyes}{combination_n_maybeno}{combination_n_no}"
            combination_samples = ((aggregate_df["yes"] == combination_n_yes)
                                   & (aggregate_df["maybeyes"] == combination_n_maybeyes)
                                   & (aggregate_df["maybeno"] == combination_n_maybeno)
                                   & (aggregate_df["no"] == combination_n_no)).sum()
            percentage_samples = 100 * combination_samples/len(aggregate_df)
            combination_rows.append([r, combination_n_yes, combination_n_maybeyes, combination_n_maybeno,
                                     combination_n_no, combination_label, combination_samples, percentage_samples])
    combination_df = pd.DataFrame(combination_rows, columns=["n", "yes", "maybeyes", "maybeno", "no", "combination",
                                                             "samples", "percentage"])

    # save the files
    dataset_df.to_csv(annotation_file, index=False)
    aggregate_df.to_csv(aggregate_file, index=False)
    combination_df.to_csv(combination_file, index=False)

if __name__ == '__main__':
    app.run(compile_annotations)