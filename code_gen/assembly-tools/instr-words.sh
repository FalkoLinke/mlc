#!/bin/bash


if [ $# -ne 1 ]; then
    echo "Usage: $0 <file.s>"
    echo ""
    echo "Assemble the mnemonics in file.s and print the instruction words to stdout."
    echo ""
    exit 1
fi

input_file=$1
OBJCOPY=/opt/homebrew/opt/binutils/bin/objcopy

clang++ -c -O0 -x assembler-with-cpp -march=armv9.2-a+nosve+sme "$input_file" -o tmp.o
if [ $? -ne 0 ]; then
    echo "Could not assemble file."
    exit 2
fi

"$OBJCOPY" -O binary -j .text tmp.o tmp.bin
if [ $? -ne 0 ]; then
    echo "Could not extract instruction words."
    rm tmp.o
    exit 3
fi

xxd -e -c 4 tmp.bin | sed 's/^[0-9,a-f,A-F]*\:[ ]//' | sed 's/^\(.\{8\}\).*/\1/'

rm tmp.o
rm tmp.bin




