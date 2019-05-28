#!/bin/bash
for i in `seq 1 10`;
do
    python3 -W ignore catalog_handler.py 10 $i\e-5 -w -n
done
