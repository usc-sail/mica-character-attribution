import os
import bs4
import csv
import re
import requests
import pandas as pd
import unidecode
import collections
import time
from bs4 import BeautifulSoup as bs

from absl import app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_string("data_dir", default=None, help="data directory", required=True)
flags.DEFINE_string("character_tropes", default="10-crawling/movie-character-tropes.csv",
                    help="csv file of character and their tropes")
flags.DEFINE_string("tropes_dir", default="10-crawling/trope-pages", help="directory of trope pages")
flags.DEFINE_string("tropes", default="10-crawling/tropes.csv", help="csv file of trope definitions")

def crawl_trope(url):
    response = requests.get(url)
    content = response.content
    soup = bs(content, features="html.parser")

    main_element = soup.find("div", attrs={"id":"main-article"})
    definition = []
    for child_element in main_element.children:
        if (not isinstance(child_element, bs4.NavigableString) 
            and (child_element.name == "hr" or child_element.find("hr") is not None)):
            break
        if child_element.name == "p" and child_element.find("div") is None:
            a_elements = child_element.find_all("a")
            for a_element in a_elements:
                if "href" in a_element.attrs:
                    trope_url = a_element.attrs["href"]
                    if "Main/" in trope_url:
                        a_element.string = f"<trope value={trope_url}>{a_element.text}</trope>"
            text = child_element.text.strip()
            text = re.sub(r"\s+", " ", text)
            if text:
                definition.append(text)
            if child_element.find("hr") is not None:
                break
    definition = "\n".join(definition)
    definition = unidecode.unidecode(definition, errors="ignore")
    return definition, content

def crawl_trope_pages(_):
    character_tropes_file = os.path.join(FLAGS.data_dir, FLAGS.character_tropes)
    tropes_file = os.path.join(FLAGS.data_dir, FLAGS.tropes)
    tropes_dir = os.path.join(FLAGS.data_dir, FLAGS.tropes_dir)

    tropes_processed = set()
    if os.path.exists(tropes_file):
        tropes_df = pd.read_csv(tropes_file, index_col=None)
        tropes_processed = set(tropes_df["trope"].tolist())
    else:
        with open(tropes_file, "w") as fw:
            fw.write("trope,definition,linked-tropes\n")
        os.makedirs(tropes_dir, exist_ok=True)

    character_tropes_df = pd.read_csv(character_tropes_file, index_col=None)
    tropes_list = character_tropes_df["trope"].str.split("/").str[-1].tolist()
    tropes_dist = collections.Counter(tropes_list).items()
    tropes_dist = sorted(tropes_dist, key=lambda item: item[1], reverse=True)
    print(f"{len(tropes_dist)} tropes, {len(set(tropes_processed))} tropes already processed")
    print()

    cumulative_sum, total = 0, len(character_tropes_df)
    with open(tropes_file, "a") as fw:
        writer = csv.writer(fw, delimiter=",")
        for i, (trope, count) in enumerate(tropes_dist):
            cumulative_sum += count
            percentage = 100*cumulative_sum/total
            if trope in tropes_processed:
                print(f"{i + 1:6d}/{len(tropes_dist)}. {trope:30s} ({percentage:.1f}%) ... already processed")
            else:
                trope_url = "https://tvtropes.org/pmwiki/pmwiki.php/Main/" + trope
                definition, content = crawl_trope(trope_url)
                content_file = os.path.join(tropes_dir, trope + ".html")
                with open(content_file, "wb") as fb:
                    fb.write(content)
                soup = bs(definition, features="html.parser")
                linked_tropes = set()
                for trope_element in soup.find_all("trope"):
                    linked_tropes.add(trope_element["value"].split("/")[-1])
                linked_tropes = ";".join(sorted(linked_tropes))
                writer.writerow([trope, definition, linked_tropes])
                print(f"{i + 1:6d}/{len(tropes_dist)}. {trope:30s} ({percentage:.1f}%)")
                time.sleep(0.5)

    tropes_df = pd.read_csv(tropes_file, index_col=None)
    tropes_df.to_csv(tropes_file, index=False)

if __name__ == '__main__':
    app.run(crawl_trope_pages)