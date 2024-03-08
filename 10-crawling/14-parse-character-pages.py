import os
import bs4
import tqdm
import pandas as pd
from bs4 import BeautifulSoup as bs

from absl import app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_string("data_dir", default=None, help="data directory", required=True)
flags.DEFINE_string("characters", default="10-crawling/movie-linked-characters.csv", 
                    help="csv of imdb-id, title, year, movie-url, rank, character-url")
flags.DEFINE_string("urls_dir", default="10-crawling/linked-character-pages", 
                    help="directory of character page html")
flags.DEFINE_string("tropes", default="10-crawling/movie-character-tropes.csv", 
                    help="csv of character-url, character name, trope, and content-text")

def istag(element):
    return isinstance(element, bs4.element.Tag)

def isheading1(element):
    return istag(element) and element.name == "h1"

def isheading2(element):
    return istag(element) and element.name == "h2"

def isheading3(element):
    return istag(element) and element.name == "h3"

def isfolderlabel(element):
    return (istag(element) and element.name == "div" and "folderlabel" in element.get_attribute_list("class")
            and "open/close all folders" not in element.text)

def isfolder(element):
    return istag(element) and element.name == "div" and "folder" in element.get_attribute_list("class")

def islist(element):
    return istag(element) and element.name == "ul"

def gettrope(list_element):
    a_element = list_element.find("a")
    if a_element is not None and "Main/" in a_element.attrs["href"]:
        url = a_element.attrs["href"]
        content = list_element.text.strip()
        link_text = a_element.text
        if content.startswith(link_text + ":"):
            content = content[len(link_text) + 1:].strip()
        return url, content

def parse_character_tropes(soup):
    main_element = soup.find("div", attrs={"id":"main-article"})
    character = [None, None, None, None, None, None] # h1/h2/h3/div[@class=folderlabel]/h2/h3
    depth = 0

    for element in main_element.children:
        content = element.text.strip()
        if isheading1(element):
            character[0] = content
            depth = 1
        elif isheading2(element):
            character[1] = content
            depth = 2
        elif isheading3(element) and not ":" in content:
            character[2] = content
            depth = 3
        elif isfolderlabel(element):
            character[3] = content
            depth = 4
        elif isfolder(element):
            for element2 in element.children:
                content2 = element2.text.strip()
                if isheading2(element2):
                    character[4] = content2
                    depth = 5
                elif isheading3(element2) and not ":" in content2:
                    character[5] = content2
                    depth = 6
                elif islist(element2):
                    for list_element in element2.find_all("li"):
                        trope = gettrope(list_element)
                        if trope is not None:
                            trope_url, trope_content = trope
                            character_content = "->".join(x for x in character[:depth] if x is not None)
                            yield character_content, trope_url, trope_content

def write_character_tropes(_):
    urls_file = os.path.join(FLAGS.data_dir, FLAGS.characters)
    urls_dir = os.path.join(FLAGS.data_dir, FLAGS.urls_dir)
    tropes_file = os.path.join(FLAGS.data_dir, FLAGS.tropes)

    urls_df = pd.read_csv(urls_file, index_col=None, dtype={"imdb-id": str})
    character_urls = urls_df["character-url"].unique()
    character_tropes = []

    for character_url in tqdm.tqdm(character_urls):
        filename = os.path.join(urls_dir, character_url.split("/")[-1] + ".html")
        with open(filename) as fr:
            content = fr.read()
        soup = bs(content, features="html.parser")
        for character, trope_url, trope_text in parse_character_tropes(soup):
            character_tropes.append([character_url, character, trope_url, trope_text])

    character_tropes_df = pd.DataFrame(character_tropes, 
                                       columns=["character-url", "character", "trope", "content-text"])
    character_tropes_df.to_csv(tropes_file, index=False)

if __name__ == '__main__':
    app.run(write_character_tropes)