#!/bin/bash
set -e
set -o pipefail

# Find our own location.
BINDIR=$(dirname "$(readlink -f "$(type -P $0 || echo $0)")")

slurm_job() {
    INSTANCE=$1
    OUTPUT=$2
    nruns=$3
    shift 2
    JOBNAME=${INSTANCE}-$counter
    # FIXME: "sbatch <<EOF" should be enough
    # FC: it does not work
    cat <<EOF > kk.sh
#!/usr/bin/env bash
# The name to show in queue lists for this job:
#SBATCH -J $JOBNAME
#SBATCH --array=1-$nruns
# Number of desired cpus:
#SBATCH --cpus-per-task=$N_SLURM_CPUS

# Amount of RAM needed for this job:
#SBATCH --mem=2gb

# The time the job will be running:
#SBATCH --time=50:00:00

# To use GPUs you have to request them:
##SBATCH --gres=gpu:1
#SBATCH --constraint=cal

# Set output and error files
#SBATCH --error=$OUTDIR/${JOBNAME}_%J.stderr
#SBATCH --output=$OUTDIR/${JOBNAME}_%J.stdout

# To load some software (you can show the list with 'module avail'):

run=\$SLURM_ARRAY_TASK_ID
COMMAND="python3 target-runner-exhaustiveexploration.py --output ${OUTPUT}-r\$run --budget $LIMIT $INSTANCE \$(( (\$run-1)*$LIMIT )) $INSTANCE"
echo "running: \$COMMAND"
\$COMMAND
EOF
sbatch kk.sh
rm kk.sh
}


factorial() {
num=$1
fact=1
while [ $num -gt 1 ]; do
  fact=$((fact * num))  #fact = fact * num
  num=$((num - 1))      #num = num - 1
done
echo $fact
}


#LAUNCHER=qsub_job
OUTDIR="$SCRATCH/asteroides"
N_SLURM_CPUS=1
LAUNCHER=slurm_job

#OUTDIR="."
#N_LOCAL_CPUS=4
#LAUNCHER=launch_local

nruns=1
LIMIT=1000
INSTANCES=""
for n in $(seq 10 10); do
    #seed=42
    for seed in 42 73 7 11 13 17 19 23 29 31; do
        INSTANCES="$INSTANCES arp_${n}_${seed}"
    done
done
#INSTANCES="arp_5_42"
# INSTANCES="
# arp_10_42
# #arp_15_42
# #arp_20_42
# "
# Filter out
#INSTANCES=$(echo "$INSTANCES" | grep -v '#' | tr '\n' ' ')
#echo $INSTANCES
#exit 0

#budget="400"
#eval_ranks="0 1"
#eval_ranks=1
#eval_ranks=0

# Actually, 10**budgetGA
#budgetGA=4

#m_ini=10
#inits="random"
#inits="maxmindist"
#inits="maxmindist greedy_euclidean"
#inits="greedy_euclidean"
#distances="kendall"
#learning="exp"
#sampling='log'

counter=0
for instance in $INSTANCES; do
    counter=$((counter+1))
    numbers=${instance#*_}
    n=${numbers%_*}
    f=$(factorial $n)
    nruns=$(( ($f / $LIMIT) +1 ))
    RESULTS="$OUTDIR/results"
    mkdir -p "$RESULTS"
    $LAUNCHER $instance "${RESULTS}/$instance" $nruns
done

