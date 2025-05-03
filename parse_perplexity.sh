#!/bin/bash

if [ "$#" -ne 1 ]; then
	echo "Usage: $0 <log_file>"
	exit 1
fi

cat "$1" | grep perplexity | sed -E 's/^[^0-9\-]*(-?[0-9]+(\.[0-9]+)?).*/\1/'
