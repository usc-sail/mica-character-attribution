"""Get character picture urls from imdb"""
import os
import re
import tqdm
import random
import requests
import pandas as pd
from IPython.display import display
from bs4 import BeautifulSoup as bs

from absl import app
from absl import flags

random.seed(2024)
FLAGS = flags.FLAGS
flags.DEFINE_string("data_dir", default=None, help="data directory", required=True)
flags.DEFINE_string("introductions_file", default="40-crowdsource/character-introductions.csv",
                    help="csv file containing fields for character and imdb character urls")
flags.DEFINE_string("urls_file", default="40-crowdsource/character-picture-urls.csv",
                    help="output csv file with additional fields for picture urls")
flags.DEFINE_string("error_urls_file", default="40-crowdsource/error-character-urls.txt",
                    help="txt file containing error character urls")

def crawl_imdb_pictures(_):
    data_dir = FLAGS.data_dir
    introductions_file = os.path.join(data_dir, FLAGS.introductions_file)
    urls_file = os.path.join(data_dir, FLAGS.urls_file)
    error_urls_file = os.path.join(data_dir, FLAGS.error_urls_file)
    introductions_df = pd.read_csv(introductions_file, index_col=None)

    rows = []
    error_urls = []
    headers={"User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                            "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36")}
    for cid, df in tqdm.tqdm(introductions_df.groupby("character"), desc="crawl character pictures", unit="character",
                             total=introductions_df["character"].unique().size):
        character_urls = set(df["imdb-character-urls"].values[0].split(";"))
        all_picture_urls = set()
        for url in character_urls:
            picture_urls = set()
            try:
                response = requests.get(url)
                soup = bs(response.content, features="html.parser")
                a_elements = (soup.find("div", {"class": "titlecharacters-image-grid"})
                              .find_all("a", {"class": "titlecharacters-image-grid__thumbnail-link"}))
                random.shuffle(a_elements)
                for a_element in a_elements:
                    try:
                        img_url = a_element.attrs["href"]
                        response = requests.get("https://imdb.com" + img_url, headers=headers)
                        img_id = re.findall(r"rm\d+", img_url)[0]
                        soup = bs(response.content, features="html.parser")
                        img_element = soup.find("img", {"data-image-id": f"{img_id}-curr"})
                        picture_urls.add(img_element.attrs["src"])
                        if len(picture_urls) == 3:
                            break
                    except Exception:
                        pass
            except Exception:
                error_urls.append(f"{cid} {url}")
            all_picture_urls.update(picture_urls)
        picture_urls = list(all_picture_urls)
        random.shuffle(picture_urls)
        if len(picture_urls) == 0:
            row = ["", "", ""]
        elif len(picture_urls) == 1:
            row = [picture_urls[0], "", ""]
        elif len(picture_urls) == 2:
            row = [picture_urls[0], picture_urls[1], ""]
        else:
            row = picture_urls[:3]
        row = [cid] + row
        rows.append(row)

    with open(error_urls_file, "w") as fw:
        fw.write("\n".join(error_urls))

    urls_df = pd.DataFrame(rows, columns=["character", "picture-1", "picture-2", "picture-3"])
    urls_df.to_csv(urls_file, index=False)

if __name__ == '__main__':
    app.run(crawl_imdb_pictures)