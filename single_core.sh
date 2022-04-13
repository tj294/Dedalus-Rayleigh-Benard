#!/bin/bash
python3 rb_convect.py -o $1;

python3 merge.py $1analysis/ --cleanup

python3 merge.py $1snapshots/ --cleanup

rm -r __pycache__
