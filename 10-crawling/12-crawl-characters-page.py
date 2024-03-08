import os
import time
import requests
import pandas as pd
from bs4 import BeautifulSoup as bs
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as ec

from absl import app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_string("data_dir", default=None, help="data directory", required=True)
flags.DEFINE_string("urls", default="10-crawling/movie-urls.csv",
                    help="csv file of movie imdb-ids, name, year, and url of the tvtrope page")
flags.DEFINE_string("characters", default="10-crawling/movie-characters.csv",
                    help=("csv file of movie imdb-ids, name, year, url of the tvtrope page, and url of the character"
                          " page"))
flags.DEFINE_string("characters_dir", default="10-crawling/character-pages",
                    help="directory of main character pages of the movies")
flags.DEFINE_bool("captcha", default=False, help="wait 10 seconds to complete captcha")

def crawl_character_page(_):
    urls_file = os.path.join(FLAGS.data_dir, FLAGS.urls)
    characters_file = os.path.join(FLAGS.data_dir, FLAGS.characters)
    characters_dir = os.path.join(FLAGS.data_dir, FLAGS.characters_dir)
    movie_df = pd.read_csv(urls_file, dtype={"imdb-id": str})

    n = 0
    if os.path.exists(characters_file):
        with open(characters_file, "rb") as fr:
            n = sum(1 for _ in fr) - 1
    else:
        with open(characters_file, "w") as fw:
            fw.write("imdb-id,title,year,url,character-url\n")

    options = webdriver.FirefoxOptions()
    driver = webdriver.Firefox(options=options)

    with open(characters_file, "a") as fw:
        for i, row in movie_df.iloc[n:].iterrows():
            imdb_id, title, year, url = row.values
            if pd.notna(url):
                response = requests.get(url)
                soup = bs(response.content, features="html.parser")
                assert soup.find("div", attrs={"id":"main-content"}) is not None, (
                    f"something went wrong retrieving {url}")
                character_element = soup.find("a", attrs={"title":"The Characters page"})
                if character_element is not None:
                    character_url = "https://tvtropes.org" + character_element.attrs["href"]
                    driver.get(character_url)
                    if FLAGS.captcha:
                        time.sleep(10)
                    WebDriverWait(driver, 5).until(
                        ec.presence_of_element_located((By.XPATH, "//div[@id='main-content']")))
                    content = driver.page_source
                    content_file = os.path.join(characters_dir, f"{imdb_id}.html")
                    with open(content_file, "w") as ff:
                        ff.write(content)
                    print(f"{i + 1:4d}. {title:30s} {character_url:50s}")
                    fw.write(f"{imdb_id},\"{title}\",{year},{url},{character_url}\n")
                else:
                    print(f"{i + 1:4d}. {title:30s} {'no character page':50s}")
                    fw.write(f"{imdb_id},\"{title}\",{year},{url},\n")
            else:
                print(f"{i + 1:4d}. {title:30s} {'no url':50s}")
                fw.write(f"{imdb_id},\"{title}\",{year},,\n")

    driver.close()

if __name__ == '__main__':
    app.run(crawl_character_page)