#!/bin/bash

CODE_PATH=$HOME/git
OUT_PATH_BASE="/storage/groups/ml01/workspace/${USER}/ncem"
GS_PATH="${OUT_PATH_BASE}/grid_searches/"
DATA_PATH="/storage/groups/ml01/workspace/${USER}/ncem/data"

MODEL_CLASS="DECONVOLUTION"
DATA_SET="destvi_lympnode"
OPTIMIZER="ADAM"
DOMAIN_TYPE="PATIENT"
LR_KEYS=("1")
L1_KEY=("1")
L2_KEYS=("1")
BATCH_SIZE=("S")
N_RINGS_KEYS=("1")
N_EVAL_KEYS=("100")

GS_KEY="$(date '+%y%m%d')c_${MODEL_CLASS}_${DOMAIN_TYPE}_${DATA_SET}"
OUT_PATH=${GS_PATH}/${GS_KEY}

# dummy values for this model class have hard-encoded default values in this grid search

rm -rf ${OUT_PATH}/jobs
rm -rf ${OUT_PATH}/logs
rm -rf ${OUT_PATH}/results
mkdir -p ${OUT_PATH}/jobs
mkdir -p ${OUT_PATH}/logs
mkdir -p ${OUT_PATH}/results

for rd in ${N_RINGS_KEYS[@]}; do
    for l1 in ${L1_KEY[@]}; do
        for bs in ${BATCH_SIZE[@]}; do
            sleep 0.1
                job_file="${OUT_PATH}/jobs/run_${MODEL_CLASS}_${DATA_SET}_${OPTIMIZER}_${LR_KEYS}_${l1}_${L2_KEYS}_${bs}_${rd}_${N_EVAL_KEYS}_${GS_KEY}.cmd"
                echo "#!/bin/bash
#SBATCH -J ${MODEL_CLASS}_${DATA_SET}_${OPTIMIZER}_${LR_KEYS}_${l1}_${L2_KEYS}_${bs}_${rd}_${N_EVAL_KEYS}_${GS_KEY}
#SBATCH -o ${OUT_PATH}/jobs/run_${MODEL_CLASS}_${DATA_SET}_${OPTIMIZER}_${LR_KEYS}_${l1}_${L2_KEYS}_${bs}_${rd}_${N_EVAL_KEYS}_${GS_KEY}.out
#SBATCH -e ${OUT_PATH}/jobs/run_${MODEL_CLASS}_${DATA_SET}_${OPTIMIZER}_${LR_KEYS}_${l1}_${L2_KEYS}_${bs}_${rd}_${N_EVAL_KEYS}_${GS_KEY}.err
#SBATCH -p gpu_p
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH -t 2-00:00:00
#SBATCH --mem=50G
#SBATCH -c 4
#SBATCH --nice=1000
#SBATCH --nodelist=supergpu05

source "$HOME"/.bashrc
conda activate ncem
python3 ${CODE_PATH}/ncem_benchmarks/scripts/train_script_linear.py ${DATA_SET} ${OPTIMIZER} ${DOMAIN_TYPE} ${LR_KEYS} ${l1} ${L2_KEYS} ${bs} ${rd} ${N_EVAL_KEYS} ${MODEL_CLASS} ${GS_KEY} ${DATA_PATH} ${OUT_PATH}
" > ${job_file}
            sbatch $job_file
        done
    done
done