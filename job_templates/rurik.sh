#!/bin/sh
#SBATCH -o out
#SBATCH -p cpu
#SBATCH -J @JOBNAME
#SBATCH -t 48:00:00
#SBATCH -N 1
#SBATCH -n @NPROCSHARED
$GAUSS_EXEDIR/g16 <  @INPUT  > @OUTFILE