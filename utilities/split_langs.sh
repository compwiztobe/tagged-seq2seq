#!/bin/bash

INFILE="$1"
SRC_LANG="$2"
TRG_LANG="$3"

grep \(src\) "$INFILE" | sed -E "s/\(src\)=\"[0-9]+\">//g" > "$INFILE.$SRC_LANG"
grep \(trg\) "$INFILE" | sed -E "s/\(trg\)=\"[0-9]+\">//g" > "$INFILE.$TRG_LANG"
