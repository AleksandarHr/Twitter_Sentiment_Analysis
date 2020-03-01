#!/bin/bash

# Note that this script uses GNU-style sed. On Mac OS, you are required to first
#    brew install gnu-sed --with-default-names
cat ../Data/train_pos.txt ../Data/train_neg.txt | sed "s/ /\n/g" | grep -v "^\s*$" | sort | uniq -c > ../Data/vocab.txt
