#!/bin/bash

if [ $# -ne 1 ]; then
    echo "Usage: $0 <template>"
    echo ""
    echo "Prints the template for each line in stdin, replacing"
    echo 'any occurrence of "\1" with the current line.'
    echo ""
    exit 1
fi

template=$1

sed 's/\(.*\)/'"$1"'/'

