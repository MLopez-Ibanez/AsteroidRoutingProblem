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

launch_local() {
    ALGO=$1
    OUTPUT=$2
    shift 2
    parallel -j $N_LOCAL_CPUS --verbose ${BINDIR}/target-runner-${ALGO}.py $ALGO $counter-$$-r{} {} $@ --output ${OUTPUT}-r{} ::: $(seq 1 $nruns)
}

OUTDIR="$HOME/scratch"
N_LOCAL_CPUS=2
#LAUNCHER=qsub_job
LAUNCHER=launch_local

nruns=5

INSTANCES="
arp_10_42
arp_15_42
arp_20_42
"
# Filter out
INSTANCES=$(echo "$INSTANCES" | grep -v '#' | tr '\n' ' ')

budget="100 200 500 1000"
eval_ranks="0 1"
# eval_ranks=1
#eval_ranks=0

# Actually, 10**budgetGA
budgetGA=4

m_ini=10
#init="random"
init="maxmindist"
distances="kendall"
#learning="exp"
#sampling='log'

counter=0
for m in $budget; do
for er in $eval_ranks; do
for distance in $distances; do
for instance in $INSTANCES; do
    counter=$((counter+1))
    RESULTS="$OUTDIR/results/m${m}-er${er}/$instance"
    mkdir -p "$RESULTS"
    #-learn_${learning}-samp_${sampling}"
    $LAUNCHER umm "${RESULTS}/umm-${init}" $instance --m_ini $m_ini --budget $m --init $init --eval_ranks $er 
    #--learning $learning --sampling $sampling --distance $distance
    $LAUNCHER cego "${RESULTS}/cego" $instance --m_ini $m_ini --budgetGA $budgetGA --budget $m --eval_ranks $er
done
done
done
done
