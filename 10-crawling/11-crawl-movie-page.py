"""Crawl TvTrope movie pages.
I was only able to run this program on my local machine and without headless mode.
The script gets stuck while running on remote machines in headless mode.
Probable reason is that the TvTrope website periodically asks if you are a robot or not, and asks you to solve a 
captcha. You need to run it in non-headless mode so you can continuously monitor the program on a display.
"""

import os
import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions

from absl import app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_string("data_dir", default=None, help="data directory", required=True)
flags.DEFINE_string("movies", default="10-crawling/movie-names.csv", 
                    help="csv file containing movie ids, names and years")
flags.DEFINE_string("urls", default="10-crawling/movie-urls.csv", 
                    help="csv file where movie ids, names, years, and trope urls will be written")
flags.DEFINE_bool("captcha", default=False, help="Wait for 10 secs to solve captcha")

def crawl_trope_movie_pages(_):
    movies_file = os.path.join(FLAGS.data_dir, FLAGS.movies)
    url_file = os.path.join(FLAGS.data_dir, FLAGS.urls)

    # create webdriver
    options = webdriver.FirefoxOptions()
    # options.add_argument("--headless") # running in headless mode does not work
    driver = webdriver.Firefox(options=options)

    # navigate to tvtropes.org
    trope_url = "https://tvtropes.org/"
    print(f"navigate to {trope_url}")
    driver.get(trope_url)
    print("done\n")

    # xpath for search box and first result item
    search_query = "//div[@id='search-box']//input[@name='q']"
    result_query = "//div[@class='gsc-webResult gsc-result']//a[@class='gs-title']"

    # read movie names and years
    movie_df = pd.read_csv(movies_file, index_col=None)

    # find number of movies already processed
    n = 0
    if os.path.exists(url_file):
        url_df = pd.read_csv(url_file, index_col=None, header=None)
        n += len(url_df)

    # array of booleans indicating whether the search was successful
    running_status = []

    # loop over movie names and years
    with open(url_file, "a") as fw:
        for i, row in movie_df.iloc[n:].iterrows():
            imdb_id, title, year = row["imdb-id"], row["title"], row["year"]

            try:
                query = f"{title} {year} film"
                search_element = driver.find_element(By.XPATH, search_query)
                search_element.clear()
                search_element.send_keys(query)
                search_element.send_keys(Keys.ENTER)
                if i == n and FLAGS.captcha:
                    time.sleep(10)
                else:
                    time.sleep(2)

                WebDriverWait(driver, 5).until(expected_conditions.text_to_be_present_in_element(
                    (By.XPATH, result_query), title))
                result_element = driver.find_element(By.XPATH, result_query)
                url = result_element.get_attribute("href")

                print(f"{i + 1:4d}. {title:30s} {year} {url}")
                fw.write(f"{imdb_id},\"{title}\",{year},{url}\n")

                running_status.append(True)
            except Exception:
                print(f"{i + 1:4d}. {title:30s} {year} ERROR")
                fw.write(f"\"{title}\",{year},\n")
                running_status.append(False)

            if len(running_status) >= 5 and not any(running_status[-5:]):
                print("terminating program because last five searches failed (website might be asking for captcha)")
                print(f"delete the last five rows from {url_file} and rerun the program with --captcha set")
                break

    # add header
    url_df = pd.read_csv(url_file, index_col=None, header=None, dtype={"0": str})
    url_df.columns = ["imdb-id", "title", "year", "url"]
    url_df.to_csv(url_file, index=False)

    # close webdriver
    driver.close()

if __name__ == '__main__':
    app.run(crawl_trope_movie_pages)