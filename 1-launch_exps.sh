#!/bin/bash
set -e
set -o pipefail

# Find our own location.
BINDIR=$(dirname "$(readlink -f "$(type -P $0 || echo $0)")")
# This function launches one job $1 is the job name, the other arguments is the job to submit.
qsub_job() {
    PARALLEL_ENV=smp.pe
    # We would like to use $BASHPID here, but OS X version of bash does not
    # support it.
    ALGO=$1
    OUTPUT=$2
    shift 2
    JOBNAME=${ALGO}-$counter-$$
    qsub -v PATH <<EOF
#!/bin/bash --login
#$ -t 1-$nruns
#$ -N $JOBNAME
# -pe $PARALLEL_ENV $NB_PARALLEL_PROCESS 
#$ -l ivybridge
#$ -M manuel.lopez-ibanez@manchester.ac.uk
#$ -m ase
#      b     Mail is sent at the beginning of the job.
#      e     Mail is sent at the end of the job.
#      a     Mail is sent when the job is aborted or rescheduled.
#      s     Mail is sent when the job is suspended.
#
#$ -o $OUTDIR/${JOBNAME}.stdout
#$ -j y
#$ -cwd
module load apps/anaconda3
run=\$SGE_TASK_ID
echo "running: ${BINDIR}/target-runner-${ALGO}.py $ALGO $counter-$$-r\$run \$run $@ --output ${OUTPUT}-r\$run"
${BINDIR}/target-runner-${ALGO}.py $ALGO $counter-$$-r\$run \$run $@ --output "${OUTPUT}-r\$run"
EOF
}

slurm_job() {
    ALGO=$1
    OUTPUT=$2
    shift 2
    JOBNAME=${ALGO}-$counter-$$
    # FIXME: "sbatch <<EOF" should be enough
    # FC: it does not work
    sbatch <(cat <<EOF
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
module load R/4.1.0_sin
module load python/3.8.8
export R_LIBS="$HOME/asteroides/R_packages"

run=\$SLURM_ARRAY_TASK_ID
echo "running: ${BINDIR}/target-runner-${ALGO}.py $ALGO $counter-$$-r\$run \$run $@ --output ${OUTPUT}-r\$run"
python3 ${BINDIR}/target-runner-${ALGO}.py $ALGO $counter-$$-r\$run \$run $@ --output "${OUTPUT}-r\$run"
EOF
)
}

launch_local() {
    ALGO=$1
    OUTPUT=$2
    shift 2
    for run in $(seq 1 $nruns); do
        parallel --semaphore -j $N_LOCAL_CPUS --verbose ${BINDIR}/target-runner-${ALGO}.py $ALGO $counter-$$-r$run $run $@ --output ${OUTPUT}-r$run
    done
}

#LAUNCHER=qsub_job
# OUTDIR="$SCRATCH/asteroides"
# N_SLURM_CPUS=1
# LAUNCHER=slurm_job

OUTDIR="."
N_LOCAL_CPUS=4
LAUNCHER=launch_local

nruns=30

INSTANCES=""
for n in $(seq 10 5 30); do
    for seed in 42 73; do
        INSTANCES="$INSTANCES arp_${n}_${seed}"
    done
done
# INSTANCES="
# arp_10_42
# #arp_15_42
# #arp_20_42
# "
# Filter out
INSTANCES=$(echo "$INSTANCES" | grep -v '#' | tr '\n' ' ')
#echo $INSTANCES
#exit 0

budget="400"
eval_ranks="0 1"
#eval_ranks=1
eval_ranks=0

# Actually, 10**budgetGA
budgetGA=4

m_ini=10
#inits="random"
#inits="maxmindist"
inits="maxmindist greedy_euclidean"
#inits="greedy_euclidean"
distances="kendall"
#learning="exp"
#sampling='log'

counter=0
for m in $budget; do
for er in $eval_ranks; do
for distance in $distances; do
for init in $inits; do
for instance in $INSTANCES; do
    counter=$((counter+1))
    RESULTS="$OUTDIR/results/m${m}-er${er}/$instance"
    mkdir -p "$RESULTS"
    # #-learn_${learning}-samp_${sampling}"
    # $LAUNCHER umm "${RESULTS}/umm-${init}" $instance --m_ini $m_ini --budget $m --init $init --eval_ranks $er 
    # #--learning $learning --sampling $sampling --distance $distance
    #$LAUNCHER cego "${RESULTS}/cego-${init}" $instance --m_ini $m_ini --budgetGA $budgetGA --budget $m --init $init --eval_ranks $er
done
done
done
done
done

for m in $budget; do
for instance in $INSTANCES; do
    counter=$((counter+1))
    RESULTS="$OUTDIR/results/m${m}/$instance"
    mkdir -p "$RESULTS"
    $LAUNCHER randomsearch "${RESULTS}/randomsearch" $instance
done
done

# We only do 1 run for greedy
nruns=1
distances="euclidean energy"
for instance in $INSTANCES; do
for distance in $distances; do
    counter=$((counter+1))
    RESULTS="$OUTDIR/results/m1-er0/$instance"
    mkdir -p "$RESULTS"
    # $LAUNCHER greedynn "${RESULTS}/greedynn-${distance}" $instance --distance $distance
done
done

