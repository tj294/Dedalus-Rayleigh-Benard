#!/bin/bash
python3 rb_convect.py -o $1;

python3 merge.py $1/analysis/ --cleanup

python3 merge.py $1/snapshots/ --cleanup

python3 analysis.py -i $1 -k

python3 analysis.py -i $1 -t

python3 analysis.py -i $1 -f

rm -r __pycache__
