# Crawling TvTropes

## Files

movie-names.csv
    fields = imdb-id, title, year

movie-urls.csv
    fields = imdb-id, title, year, url

movie-characters.csv
    fields = imdb-id, title, year, url, character-url

movie-linked-characters.csv
    fields = imdb-id, title, year, url, rank, character-url

movie-character-tropes.csv
    fields = character-url, character, trope, content-text

movie-tropes.csv
    fields = imdb-id, title, year, url, trope, content-text

tropes.csv
    fields = trope, definition, linked-tropes