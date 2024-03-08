"""Crawl tropes from a content page in tvtropes.org
The program writes rows of content-id, title, year, content-url, trope-url, content-text
Therefore, basically it adds the trope-url and content-text columns to the input urls file
Some preprocessing is required to remove the quotes from around the content-text field so the file is pandas-readable
Also, the program terminates if five successive pages returns an error
In that case, you are advised to wait few minutes and restart the program 
"""
import re
import os
import requests
from bs4 import BeautifulSoup as bs
import pandas as pd

from absl import app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_string("data_dir", default=None, help="data directory", required=True)
flags.DEFINE_string("urls", help="csv file of urls", default=None, required=True)
flags.DEFINE_string("tropes", help="csv file of extracted tropes", default=None, required=True)

def crawl_tropes(_):
    url_file = os.path.join(FLAGS.data_dir, FLAGS.urls)
    tropes_file = os.path.join(FLAGS.data_dir, FLAGS.tropes)

    with open(url_file) as fr:
        for line in fr:
            content_id_field = line.split(",")[0]
            break
    url_df = pd.read_csv(url_file, index_col=None, dtype={content_id_field: str})

    n = 0
    if os.path.exists(tropes_file):
        last_content_id = None
        with open(tropes_file, "r") as fr:
            for i, line in enumerate(fr):
                if i > 0:
                    last_content_id = line.split(",")[0]
        if last_content_id is not None:
            n = url_df.loc[url_df[content_id_field] == last_content_id].index.item() + 1
    else:
        with open(tropes_file, "w") as fw:
            fw.write(f"{content_id_field},title,year,url,trope-url,content-text\n")

    running_status = []

    with open(tropes_file, "a") as fw:
        for i, row in url_df.iloc[n:].iterrows():
            content_id, title, year, url = row.values
            if pd.notna(url):
                try:
                    response = requests.get(url, timeout=5)
                    soup = bs(response.content, features="html.parser")
                    list_elements = soup.find("div", attrs={"id": "main-article"}).find_all("li")
                    m = 0
                    for list_element in list_elements:
                        a_element = list_element.find("a")
                        if a_element is not None:
                            trope_url = a_element.attrs["href"]
                            if "Main/" in trope_url:
                                trope_text = list_element.text
                                trope_text = re.sub(r"\s+", " ", trope_text).strip()
                                fw.write(f"{content_id},\"{title}\",{year},{url},{trope_url},\"{trope_text}\"\n")
                                m += 1
                    print(f"{i + 1:4d}. {title:40s} {year} {url:100s} = {m:4d} tropes extracted")
                    running_status.append(True)
                except Exception:
                    print(f"{i + 1:4d}. {title:40s} {year} {url:100s} = ERROR")
                    running_status.append(False)
                    if len(running_status) >= 5 and not any(running_status[-5:]):
                        print("terminating program because last five crawls failed")
                        print("this happened probably because of tvtropes rate limit or bot detection scripts")
                        print("retry after some time!")
                        break
            else:
                print(f"{i + 1:4d}. {title:40s} {year} NO URL")
                running_status.append(True)

if __name__ == '__main__':
    app.run(crawl_tropes)