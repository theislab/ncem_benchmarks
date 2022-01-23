#!/bin/bash

CODE_PATH=$HOME/git
OUT_PATH_BASE="/storage/groups/ml01/workspace/${USER}/ncem"
GS_PATH="${OUT_PATH_BASE}/grid_searches/"
DATA_PATH="/storage/groups/ml01/workspace/${USER}/ncem/data"

MODEL_CLASS="ED_NCEM"
DATA_SET="destvi_lymphnode"
COND_TYPE="GCN"
OPTIMIZER="ADAM"
DOMAIN_TYPE="PATIENT"
LR_KEYS=("1")
LAT_DIM_KEYS=("1+2+3")
DR_KEYS=("1")
L1_KEY=("1")
L2_KEYS=("2")

ENC_INT_DIM_KEYS=("6")
ENC_DEPTH_KEYS=("0" "1")
DEC_INT_DIM_KEY=("5")
DEC_DEPTH_KEY=("0" "1")

COND_DEPTH_KEY=("1")
COND_DIM_KEY=("0")
COND_DR_KEY=("1")
COND_L2_KEY=("2")

BATCH_SIZE=("S")
RADIUS_KEYS=("0" "1" "5" "100")
N_EVAL_KEYS=("100")
USE_TYPE_COND="1"

GS_KEY="$(date '+%y%m%d')_${MODEL_CLASS}_${COND_TYPE}_${DOMAIN_TYPE}_${DATA_SET}"
OUT_PATH=${GS_PATH}/${GS_KEY}

# dummy values for this model class have hard-encoded default values in this grid search

rm -rf ${OUT_PATH}/jobs
rm -rf ${OUT_PATH}/logs
rm -rf ${OUT_PATH}/results
mkdir -p ${OUT_PATH}/jobs
mkdir -p ${OUT_PATH}/logs
mkdir -p ${OUT_PATH}/results

for dr in ${DR_KEYS[@]}; do
    for eid in ${ENC_INT_DIM_KEYS[@]}; do
        for ede in ${ENC_DEPTH_KEYS[@]}; do
            for did in ${DEC_INT_DIM_KEY[@]}; do
                for dde in ${DEC_DEPTH_KEY[@]}; do
                    for cde in ${COND_DEPTH_KEY[@]}; do
                        for cdi in ${COND_DIM_KEY[@]}; do
                            for cdr in ${COND_DR_KEY[@]}; do
                                for cl2 in ${COND_L2_KEY[@]}; do
                                    for rd in ${RADIUS_KEYS[@]}; do
                                        for l1 in ${L1_KEY[@]}; do
                                            for bs in ${BATCH_SIZE[@]}; do
                                                sleep 0.1
                                                    job_file="${OUT_PATH}/jobs/run_${MODEL_CLASS}_${DATA_SET}_${OPTIMIZER}_${COND_TYPE}_${LR_KEYS}_${LAT_DIM_KEYS}_${dr}_${l1}_${L2_KEYS}_${eid}_${ede}_${did}_${dde}_${cde}_${cdi}_${cdr}_${cl2}_${bs}_${rd}_${N_EVAL_KEYS}_${USE_TYPE_COND}_${GS_KEY}.cmd"
                                                    echo "#!/bin/bash
#SBATCH -J ${MODEL_CLASS}_${DATA_SET}_${OPTIMIZER}_${COND_TYPE}_${LR_KEYS}_${LAT_DIM_KEYS}_${dr}_${l1}_${L2_KEYS}_${eid}_${ede}_${did}_${dde}_${cde}_${cdi}_${cdr}_${cl2}_${bs}_${rd}_${N_EVAL_KEYS}_${USE_TYPE_COND}_${GS_KEY}
#SBATCH -o ${OUT_PATH}/jobs/run_${MODEL_CLASS}_${DATA_SET}_${OPTIMIZER}_${COND_TYPE}_${LR_KEYS}_${LAT_DIM_KEYS}_${dr}_${l1}_${L2_KEYS}_${eid}_${ede}_${did}_${dde}_${cde}_${cdi}_${cdr}_${cl2}_${bs}_${rd}_${N_EVAL_KEYS}_${USE_TYPE_COND}_${GS_KEY}.out
#SBATCH -e ${OUT_PATH}/jobs/run_${MODEL_CLASS}_${DATA_SET}_${OPTIMIZER}_${COND_TYPE}_${LR_KEYS}_${LAT_DIM_KEYS}_${dr}_${l1}_${L2_KEYS}_${eid}_${ede}_${did}_${dde}_${cde}_${cdi}_${cdr}_${cl2}_${bs}_${rd}_${N_EVAL_KEYS}_${USE_TYPE_COND}_${GS_KEY}.err
#SBATCH -p gpu_p
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH -t 2-00:00:00
#SBATCH --mem=50G
#SBATCH -c 4
#SBATCH --nice=1000

source "$HOME"/.bashrc
conda activate ncem
python3 ${CODE_PATH}/ncem_benchmarks/scripts/train_script_ed_ncem.py ${DATA_SET} ${OPTIMIZER} ${COND_TYPE} ${DOMAIN_TYPE} ${LR_KEYS} ${LAT_DIM_KEYS} ${dr} ${l1} ${L2_KEYS} ${eid} ${ede} ${did} ${dde} ${cde} ${cdi} ${cdr} ${cl2} ${bs} ${rd} ${N_EVAL_KEYS} ${USE_TYPE_COND} ${MODEL_CLASS} ${GS_KEY} ${DATA_PATH} ${OUT_PATH}
" > ${job_file}
                                                sbatch $job_file
                                            done
                                        done
                                    done
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done