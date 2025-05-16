#!/bin/bash

ANONYMIZES=("")
WORDS=(1000)
SIMS=(0.05 0.1 0.15 0.2)

for ANONYMIZE in "${ANONYMIZES[@]}"; do
    for WORD in "${WORDS[@]}"; do
        python 507-contexts.py --max_words_per_context $WORD --strategy random --${ANONYMIZE}anonymize
        python 507-contexts.py --max_words_per_context $WORD --strategy first --${ANONYMIZE}anonymize
        python 507-contexts.py --max_words_per_context $WORD --strategy last --${ANONYMIZE}anonymize
        for SIM in "${SIMS[@]}"; do
            python 507-contexts.py --max_words_per_context $WORD --strategy trope \
                --min_pos_similarity $SIM --max_neg_similarity -$SIM --${ANONYMIZE}anonymize
        done
    done
done