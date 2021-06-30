#!/bin/bash

CODE_PATH=$HOME/git
OUT_PATH_BASE="."
GS_PATH="${OUT_PATH_BASE}/grid_searches/"
DATA_PATH="."

SBATCH_P=""
SBATCH_QOS=""
SBATCH_GRES=""
SBATCH_TIME=""
SBATCH_MEM=""
SBATCH_C=""
SBATCH_NICE=""
SBATCH_EXCLUDE=""

MODEL_CLASS="INTERACTIONS"
DATA_SET="JAROSCH"
OPTIMIZER="ADAM"
DOMAIN_TYPE="PATIENT"
LR_KEYS=("1")
L1_KEY=("1" "4" "6")
L2_KEYS=("1+4+6")
BATCH_SIZE=("S")
RADIUS_KEYS=("0" "1" "2" "3" "4" "5" "6" "7")
N_EVAL_KEYS=("100")

GS_KEY="$(date '+%y%m%d')_${MODEL_CLASS}_${DOMAIN_TYPE}_${DATA_SET}"
OUT_PATH=${GS_PATH}/${GS_KEY}

# dummy values for this model class have hard-encoded default values in this grid search

rm -rf ${OUT_PATH}/jobs
rm -rf ${OUT_PATH}/logs
rm -rf ${OUT_PATH}/results
mkdir -p ${OUT_PATH}/jobs
mkdir -p ${OUT_PATH}/logs
mkdir -p ${OUT_PATH}/results

for rd in ${RADIUS_KEYS[@]}; do
    for l1 in ${L1_KEY[@]}; do
        for bs in ${BATCH_SIZE[@]}; do
            sleep 0.1
                job_file="${OUT_PATH}/jobs/run_${MODEL_CLASS}_${DATA_SET}_${OPTIMIZER}_${LR_KEYS}_${l1}_${L2_KEYS}_${bs}_${rd}_${N_EVAL_KEYS}_${GS_KEY}.cmd"
                echo "#!/bin/bash
#SBATCH -J ${MODEL_CLASS}_${DATA_SET}_${OPTIMIZER}_${LR_KEYS}_${l1}_${L2_KEYS}_${bs}_${rd}_${N_EVAL_KEYS}_${GS_KEY}
#SBATCH -o ${OUT_PATH}/jobs/run_${MODEL_CLASS}_${DATA_SET}_${OPTIMIZER}_${LR_KEYS}_${l1}_${L2_KEYS}_${bs}_${rd}_${N_EVAL_KEYS}_${GS_KEY}.out
#SBATCH -e ${OUT_PATH}/jobs/run_${MODEL_CLASS}_${DATA_SET}_${OPTIMIZER}_${LR_KEYS}_${l1}_${L2_KEYS}_${bs}_${rd}_${N_EVAL_KEYS}_${GS_KEY}.err
#SBATCH -p ${SBATCH_P}
#SBATCH -q ${SBATCH_QOS}
#SBATCH --gres=${SBATCH_GRES}
#SBATCH -t ${SBATCH_TIME}
#SBATCH --mem=${SBATCH_MEM}
#SBATCH -c ${SBATCH_C}
#SBATCH --nice=${SBATCH_NICE}
#SBATCH --exclude=${SBATCH_EXCLUDE}

source "$HOME"/.bashrc
conda activate ncem
python3 ${CODE_PATH}/ncem_benchmarks/scripts/train_script_linear.py ${DATA_SET} ${OPTIMIZER} ${DOMAIN_TYPE} ${LR_KEYS} ${l1} ${L2_KEYS} ${bs} ${rd} ${N_EVAL_KEYS} ${MODEL_CLASS} ${GS_KEY} ${DATA_PATH} ${OUT_PATH}
" > ${job_file}
            sbatch $job_file
        done
    done
done