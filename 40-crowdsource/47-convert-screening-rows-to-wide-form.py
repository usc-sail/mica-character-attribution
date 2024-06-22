"""Add category (found from gpt prompting) to movies and convert screening rows into wide format"""
import os
import re
import collections
import pandas as pd

from absl import flags
from absl import app

FLAGS = flags.FLAGS
flags.DEFINE_string("data_dir", default=None, required=True, help="data directory")
flags.DEFINE_string("screening_file", default="40-crowdsource/screening-movies.csv",
                    help="CSV file containing movie rows in long format: one movie per row")
flags.DEFINE_string("clustering_prompt_response_file", default="40-crowdsource/clusters/cluster-1.txt",
                    help="TXT file containing the output of prompting gpt to cluster movies")
flags.DEFINE_string("wide_form_screening_file", default="40-crowdsource/wide-form-screening-movies.csv",
                    help="CSV file containing movie rows in wide format: multiple movies of same category per row")

def convert_screening_rows_to_wide_form(_):
    data_dir = FLAGS.data_dir
    screening_file = os.path.join(data_dir, FLAGS.screening_file)
    clustering_prompt_response_file = os.path.join(data_dir, FLAGS.clustering_prompt_response_file)
    wide_form_screening_file = os.path.join(data_dir, FLAGS.wide_form_screening_file)

    screening_df = pd.read_csv(screening_file, index_col=None, dtype=str)
    with open(clustering_prompt_response_file) as fr:
        clustering_content = fr.read()

    cluster_to_titles = collections.defaultdict(list)
    current_cluster = None
    for match in re.finditer(r"(### (.+))|(\d+\. ([^\(]+) \(\d+\))", clustering_content):
        cluster = match.group(2)
        title = match.group(4)
        if cluster:
            current_cluster = cluster.strip()
        else:
            cluster_to_titles[current_cluster].append(title.strip())
    clustered_titles = [title for titles in cluster_to_titles.values() for title in titles]
    assert len(clustered_titles) == len(set(clustered_titles)) and set(clustered_titles) == set(screening_df["title"])

    screening_df["category"] = ""
    for category, titles in cluster_to_titles.items():
        for title in titles:
            screening_df.loc[screening_df["title"] == title, "category"] = category
    print(collections.Counter(screening_df["category"]))

    wide_rows = []
    # m = 10 # number of movies per page
    for category, category_df in screening_df.groupby("category"):
        imdb_ids = category_df["imdb-id"].tolist()
        titles = category_df["title"].tolist()
        years = category_df["year"].tolist()
        picture_urls = category_df["picture-url"].tolist()
        hq_picture_urls = category_df["picture-url-hq"].tolist()
        # n = len(category_df)
        # n_rows = n // m
        # remainder = n % m
        # if remainder >= m/2:
        #     n_rows += 1
        # for i in range(n_rows):
        #     j = i * m + m
        #     if i == n_rows - 1:
        #         j = n
        #     ix = slice(i * m, j)
        #     wide_rows.append([";".join(imdb_ids[ix]), ";".join(titles[ix]), ";".join(years[ix]),
        #                       ";".join(picture_urls[ix]), ";".join(hq_picture_urls[ix]), category])
        wide_rows.append([";".join(imdb_ids), ";".join(titles), ";".join(years), ";".join(picture_urls), 
                          ";".join(hq_picture_urls), category])

    wide_screening_df = pd.DataFrame(wide_rows, columns=["imdb-ids", "titles", "years", "picture-urls",
                                                         "picture-urls-hq", "category"])
    wide_screening_df.to_csv(wide_form_screening_file, index=False)

if __name__ == '__main__':
    app.run(convert_screening_rows_to_wide_form)