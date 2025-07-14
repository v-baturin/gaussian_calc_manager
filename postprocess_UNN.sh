#!/bin/sh
while [ ! -f "./DONE" ] && [ ! -f "./STOP" ]; do
  srun -x "node[1,78,13,120,36]" -N 1 -p gpu -t 300 db_postproc.py "$@"
done
