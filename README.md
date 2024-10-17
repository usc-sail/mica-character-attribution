# CHATTER : Character Attribution Dataset for Narrative Understanding

CHATTER is a character attribution dataset.
It contains character-trope pairs with binary labels to indicate whether the character portrayed the trope or not.

We draw the characters from Hollywood movies.
Most movies released after 1980 and covers various genres.
In total, we cover 2998 characters from 660 movies.
CHATTER contains the screenplays of the movies and maps them to their IMDb page.
The average size of the screenplay document is 25K words.
Therefore, CHATTER is a long-document dataset for character attribution.

We draw the tropes from [TVTropes](https://tvtropes.org), which is a community-driven resource similar to Wikipedia.
It contains trope annotations of literature, movies, TV shows, video games, print media and comics.
Fans and followers of any creative work identify the tropes portrayed in the story and post them on the TVTropes page for that work.
TVTropes editors and users maintain and revise the content regularly.

However, we need a more reliable dataset to serve as the test set for character attribution.
Therefore, we annotate a subset of the CHATTER dataset to create the CHATTEREVAL evaluation set.
CHATTEREVAL contains human-validated attribution labels for 1062 character-trope pairs.
We use CHATTER as the training set and CHATTEREVAL as the test set for the character attribution task.

The below table shows the data statistics of CHATTER and CHATTEREVAL.

| Dataset     | Samples | Characters | Tropes | Movies |
|-------------|---------|------------|--------|--------|
| CHATTER     | 88148   | 2998       | 13324  | 660    |
| CHATTEREVAL | 1062    | 271        | 897    | 78     |

CHATTEREVAL also contains the annotation labels of the individual raters to encourage multi-annotator modeling.

## Baselines

We develop baselines using zero-shot prompting on CHATTEREVAL for the character attribution task.
The below table shows the performance of prompting the closed-source Gemini-1.5-Flash and the open-source Llama-3.1-8B-Instruct models on the full movie script or the script segments containing the character's mentions or utterances.
We also evaluate the performance of the CHATTER dataset's labels on CHATTEREVAL.

| Input   | Model                 | Precision | Recall | F1   | weighted-F1 |
|---------|-----------------------|-----------|--------|------|-------------|
|         | Random                | 52.3      | 50.0   | 51.1 | 51.1        |
|         | CHATTER's labels      | 81.0      | 82.2   | 81.6 | 83.6        |
| Script  | Llama-3.1-8B-Instruct | 61.9      | 24.8   | 35.4 | 36.1        |
|         | Gemini-1.5-Flash      | 92.3      | 39.7   | 55.5 | 59.1        |
| Segment | Llama-3.1-8B-Instruct | 62.9      | 60.5   | 61.7 | 62.8        |
|         | Gemini-1.5-Flash      | 88.1      | 66.3   | 75.7 | 79.3        |

## Repository Organization

This repository contains the code used to curate the CHATTER and CHATTEREVAL dataset.