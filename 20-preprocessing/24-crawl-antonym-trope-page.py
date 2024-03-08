import os
import bs4
import csv
import re
import requests
import pandas as pd
import unidecode
import time
from bs4 import BeautifulSoup as bs

from absl import app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_string("data_dir", default=None, help="data directory", required=True)
flags.DEFINE_string("tropes", default="20-preprocessing/tropes-with-antonyms.csv", 
                    help="csv file of trope, definition, linked-tropes, antonym-tropes")
flags.DEFINE_string("antonym_tropes", default="20-preprocessing/antonym-tropes.csv",
                    help="csv file of trope, definition, linked-tropes")
flags.DEFINE_string("tropes_dir", default="10-crawling/trope-pages", help="directory of trope pages")

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

def crawl_antonym_trope_pages(_):
    tropes_file = os.path.join(FLAGS.data_dir, FLAGS.tropes)
    tropes_dir = os.path.join(FLAGS.data_dir, FLAGS.tropes_dir)
    antonym_tropes_file = os.path.join(FLAGS.data_dir, FLAGS.antonym_tropes)

    antonym_tropes_processed = set()
    if os.path.exists(antonym_tropes_file):
        antonym_tropes_df = pd.read_csv(antonym_tropes_file, index_col=None)
        antonym_tropes_processed = set(antonym_tropes_df["trope"].tolist())
    else:
        with open(antonym_tropes_file, "w") as fw:
            fw.write("trope,definition,linked-tropes\n")
        os.makedirs(tropes_dir, exist_ok=True)

    tropes_df = pd.read_csv(tropes_file, index_col=None)
    tropes = set(tropes_df["trope"].tolist())
    antonym_tropes = [antonym_trope for antonym_tropes in tropes_df["antonym-tropes"].dropna().str.split(";")
                      for antonym_trope in antonym_tropes]
    antonym_tropes = sorted(set(antonym_tropes).difference(tropes))
    print(f"{len(antonym_tropes)} antonym tropes, {len(antonym_tropes_processed)} antonym tropes already processed")
    print()

    with open(antonym_tropes_file, "a") as fw:
        writer = csv.writer(fw, delimiter=",")
        for i, antonym_trope in enumerate(antonym_tropes):
            if antonym_trope in antonym_tropes_processed:
                print(f"{i + 1:6d}/{len(antonym_tropes)}. {antonym_trope:30s} ... already processed")
            else:
                trope_url = "https://tvtropes.org/pmwiki/pmwiki.php/Main/" + antonym_trope
                definition, content = crawl_trope(trope_url)
                content_file = os.path.join(tropes_dir, antonym_trope + ".html")
                with open(content_file, "wb") as fb:
                    fb.write(content)
                soup = bs(definition, features="html.parser")
                linked_tropes = set()
                for trope_element in soup.find_all("trope"):
                    linked_tropes.add(trope_element["value"].split("/")[-1])
                linked_tropes = ";".join(sorted(linked_tropes))
                writer.writerow([antonym_trope, definition, linked_tropes])
                print(f"{i + 1:6d}/{len(antonym_tropes)}. {antonym_trope:30s}")
                time.sleep(0.5)

    antonym_tropes_df = pd.read_csv(antonym_tropes_file, index_col=None)
    antonym_tropes_df.to_csv(antonym_tropes_file, index=False)

if __name__ == '__main__':
    app.run(crawl_antonym_trope_pages)