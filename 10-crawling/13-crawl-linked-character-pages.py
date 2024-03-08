import os
import time
import pandas as pd
from bs4 import BeautifulSoup as bs
import shutil
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as ec

from absl import app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_string("data_dir", default=None, help="data directory", required=True)
flags.DEFINE_string("characters", default="10-crawling/movie-characters.csv",
                    help=("csv file of movie imdb-ids, names, year, tvtrope url of the movie, and tvtrope url of the"
                          " movie's characters"))
flags.DEFINE_string("characters_dir", default="10-crawling/character-pages",
                    help="directory of main character tvtrope html files of movies")
flags.DEFINE_string("lcharacters", default="10-crawling/movie-linked-characters.csv",
                    help=("csv file of movie imdb-id, title, year, tvtrope url of the movie, rank, and tvtrope url of"
                          " the movie's characters. rank can be main or link-x"))
flags.DEFINE_string("lcharacters_dir", default="10-crawling/linked-character-pages",
                    help=("directory of linked character tvtrope html files"))
flags.DEFINE_bool("captcha", default=False, help="set to wait for 10 seconds and solve captcha")

def crawl_linked_character_pages(_):
    urls_file = os.path.join(FLAGS.data_dir, FLAGS.characters)
    linked_file = os.path.join(FLAGS.data_dir, FLAGS.lcharacters)
    character_dir = os.path.join(FLAGS.data_dir, FLAGS.characters_dir)
    linked_dir = os.path.join(FLAGS.data_dir, FLAGS.lcharacters_dir)

    urls_df = pd.read_csv(urls_file, index_col=None, dtype={"imdb-id": str})
    urls_processed = set()

    if not os.path.exists(linked_file):
        with open(linked_file, "w") as fw:
            fw.write("imdb-id,title,year,url,rank,character-url\n")
    else:
        linked_df = pd.read_csv(linked_file, index_col=None, dtype={"imdb-id": str})
        if not linked_df.empty:
            last_imdb_id = linked_df["imdb-id"].values[-1]
            i = urls_df[urls_df["imdb-id"] == last_imdb_id].index.item()
            urls_df = urls_df.iloc[i:]
            urls_processed = set(linked_df["imdb-id"].unique().tolist())

    driver = webdriver.Firefox()
    captcha_done = False

    with open(linked_file, "a") as fw:
        for i, row in urls_df.iterrows():
            imdb_id, title, year, url, character_url = row.values
            if pd.notna(character_url):
                main_file = os.path.join(character_dir, f"{imdb_id}.html")
                dst_path = os.path.join(linked_dir, character_url.split("/")[-1] + ".html")
                shutil.copy2(main_file, dst_path)
                fw.write(f"{imdb_id},\"{title}\",{year},{url},main,{character_url}\n")
                print(f"{i + 1:4d}. {title:40s} copy main character page")
                urls_processed.add(character_url)

                with open(dst_path) as fr:
                    content = fr.read()
                soup = bs(content, features="html.parser")

                main_element = soup.find("div", attrs={"id":"main-article"})
                link_elements = main_element.find_all("a")
                for link_element in link_elements:
                    link_url = link_element.attrs["href"]
                    if not link_url.startswith("https://tvtropes.org"):
                        link_url = "https://tvtropes.org" + link_url
                    if "/Characters/" in link_url:
                        if link_url not in urls_processed:
                            driver.get(link_url)
                            if FLAGS.captcha and not captcha_done:
                                time.sleep(10)
                                captcha_done = True
                            WebDriverWait(driver, 5).until(
                                ec.presence_of_element_located((By.XPATH, "//div[@id='main-content']")))
                            time.sleep(1)
                            content = driver.page_source
                            dst_path = os.path.join(linked_dir, link_url.split("/")[-1] + ".html")
                            with open(dst_path, "w") as ff:
                                ff.write(content)
                            urls_processed.add(link_url)
                        fw.write(f"{imdb_id},\"{title}\",{year},{url},link-1,{link_url}\n")
                        print(f"{i + 1:4d}. {title:40s} {link_url}")

    linked_df = pd.read_csv(linked_file, index_col=None, dtype={"imdb-id": str})
    linked_df = linked_df.drop_duplicates()
    linked_df.to_csv(linked_file, index=False)
    driver.close()

if __name__ == '__main__':
    app.run(crawl_linked_character_pages)