#!/bin/bash
python 502-zeroshot-script.py --gemini_model gemini-1.5-flash --gemini_key /home/sbaruah/gemini-key.txt \
    --max_output_tokens 256 --anonymize
python 502-zeroshot-script.py --gpt_model gpt-4o-mini --gpt_key /home/sbaruah/openai-key.yml \
    --max_output_tokens 256 --anonymize