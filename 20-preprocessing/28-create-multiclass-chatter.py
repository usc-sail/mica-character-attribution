# Create the multi-class classification version of CHATTER
import collections
import os
import pandas as pd
import random
import tqdm

from absl import app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_string(name="datadir", default="/data1/sbaruah/mica-character-attribution", help="data directory")

def create_multiclass_chatter(_):
    # set file paths
    chatter_file = os.path.join(FLAGS.datadir, "CHATTER/chatter.csv")
    antonyms_file = os.path.join(FLAGS.datadir, "20-preprocessing/tropes-with-antonyms.csv")
    multiclass_chatter_file = os.path.join(FLAGS.datadir, "CHATTER/chatter-multiclass.csv")

    # read binary-class chatter -> upper-case tropes -> create new label column called "unified-label"
    # unified-label = human-label if partition == "test" else tvtrope-label
    chatter_df = pd.read_csv(chatter_file, index_col=None)
    chatter_df["trope"] = chatter_df["trope"].str.upper()
    chatter_df.drop_duplicates(("character", "trope"), inplace=True)
    chatter_df["unified-label"] = chatter_df["tvtrope-label"]
    chatter_df.loc[chatter_df["partition"] == "test", "unified-label"] = (
        chatter_df[chatter_df["partition"] == "test"]["label"].astype(int))

    # read antonyms -> upper-case tropes -> create trope to antonyms dictionary
    antonyms_df = pd.read_csv(antonyms_file, index_col=None)
    antonyms_df["trope"] = antonyms_df["trope"].str.upper()
    antonyms_df.drop_duplicates("trope", inplace=True)
    trope_to_antonym_tropes = collections.defaultdict(set)
    for _, row in antonyms_df[antonyms_df["antonym-tropes"].notna()].iterrows():
        antonym_tropes = set([trope.upper() for trope in row["antonym-tropes"].split(";")])
        antonym_tropes.discard(row["trope"])
        trope_to_antonym_tropes[row["trope"]] = antonym_tropes

    # find distribution of portrayed tropes
    header = ["character", "trope1", "trope2", "trope3", "trope4", "trope5", "label", "weight", "partition"]
    tropes_distr = collections.Counter(chatter_df.loc[chatter_df["unified-label"] == 1, "trope"])
    all_tropes = set(chatter_df[chatter_df["unified-label"] == 1]["trope"])
    multiclass_chatter_rows = []
    random.seed(2025)

    # create multiclass chatter dataset
    # for each character, find portrayed tropes, antonyms of portrayed tropes, and all other remaining tropes
    # portrayed tropes, antonyms, and remaining tropes are all mutually disjoint
    # sample two antonyms and two tropes from the remaining tropes to create a 5-way classification sample
    for character, df in tqdm.tqdm(chatter_df[chatter_df["unified-label"] == 1].groupby("character"),
                                   desc="creating multiclass samples",
                                   total=chatter_df["character"].unique().size):
        portrayed_tropes = set(df["trope"])
        antonyms_of_portrayed_tropes = set()
        for trope in portrayed_tropes:
            antonyms_of_portrayed_tropes.update(trope_to_antonym_tropes[trope])
        antonyms_of_portrayed_tropes.difference_update(portrayed_tropes)
        remaining_tropes = all_tropes.difference(portrayed_tropes).difference(antonyms_of_portrayed_tropes)
        remaining_tropes = list(remaining_tropes)
        remaining_tropes_counts = [tropes_distr[trope] for trope in remaining_tropes]
        for _, row in df.iterrows():
            antonym_tropes = list(trope_to_antonym_tropes[row["trope"]].intersection(antonyms_of_portrayed_tropes))
            antonym_tropes = random.sample(antonym_tropes, min(2, len(antonym_tropes)))
            other_tropes = []
            while len(set(other_tropes)) < 4 - len(antonym_tropes):
                other_tropes = random.sample(remaining_tropes, k=4-len(antonym_tropes), counts=remaining_tropes_counts)
            tropes = [row["trope"]] + antonym_tropes + other_tropes
            random.shuffle(tropes)
            label = tropes.index(row["trope"]) + 1
            multiclass_row = [character] + tropes + [label, row["weight"], row["partition"]]
            multiclass_chatter_rows.append(multiclass_row)
    multiclass_chatter_df = pd.DataFrame(multiclass_chatter_rows, columns=header)
    multiclass_chatter_df.to_csv(multiclass_chatter_file, index=False)

    # print data statistics
    # print distribution of samples across partitions: train, dev, and test
    # print number of ambiguous pairs, two samples constitute an ambiguous pair if they are of the same character and
    # the portrayed tropes are antonyms of each other, this does not necessarily mean the TVTrope or human annotation
    # is wrong because our method of choosing antonyms is not very precise
    partition_distr = collections.Counter(multiclass_chatter_df["partition"])
    print(f"\ndistribution of samples: {partition_distr}\n")
    ambiguous_pairs = []
    total_pairs = 0
    for _, df in multiclass_chatter_df.groupby("character"):
        n = len(df)
        tropes = [row[f"trope{row['label']}"] for _, row in df.iterrows()]
        assert len(set(tropes)) == n
        for i in range(n - 1):
            for j in range(i + 1, n):
                if tropes[j] in trope_to_antonym_tropes[tropes[i]] or tropes[i] in trope_to_antonym_tropes[tropes[j]]:
                    ambiguous_pairs.append(tuple(sorted([tropes[i], tropes[j]])))
        total_pairs += n * (n - 1) / 2
    percentage = 100 * len(ambiguous_pairs) / total_pairs
    print(f"{len(ambiguous_pairs)} ambiguous pairs ({percentage:.2f}% of all possible pairs)")
    ambiguous_pairs_distr = collections.Counter(ambiguous_pairs)
    print(f"top 20 common ambiguous trope pairs:")
    print(ambiguous_pairs_distr.most_common(20))

if __name__ == '__main__':
    app.run(create_multiclass_chatter)