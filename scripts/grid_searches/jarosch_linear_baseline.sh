#!/bin/bash

CODE_PATH=$HOME/git
OUT_PATH_BASE="/storage/groups/ml01/workspace/${USER}/tissue"
DATA_PATH_BASE="/storage/groups/ml01/workspace/${USER}/tissue/data"
PARTITION="gpu_p"

MODEL_CLASS="LINEAR_BASELINE"
DATA_SET="JAROSCH"
OPTIMIZER=("ADAM")
COND_TYPE=("MAX")
DOMAIN_TYPE=("PATIENT")
LR_KEYS=("1")
L1_KEY=("1")
L2_KEYS=("1")
USE_TYPE_COND=("1")
BATCH_SIZE=("S")
RADIUS_KEYS=("0")
TRANSFORM_KEY=("4")
N_EVAL_KEYS=("100")
SPLIT_MODE=("NODES")

GS_KEY="210520_${MODEL_CLASS}_${COND_TYPE}_${SPLIT_MODE}_${DOMAIN_TYPE}_${USE_TYPE_COND}_${DATA_SET}"
OUT_PATH=${OUT_PATH_BASE}/grid_searches_gen/${GS_KEY}

# dummy values for this model class have hard-encoded default values in this grid search

rm -rf ${OUT_PATH}/jobs
rm -rf ${OUT_PATH}/logs
rm -rf ${OUT_PATH}/results
mkdir -p ${OUT_PATH}/jobs
mkdir -p ${OUT_PATH}/logs
mkdir -p ${OUT_PATH}/results

for rd in ${RADIUS_KEYS[@]}; do
    for vl1 in ${L1_KEY[@]}; do
        for bs in ${BATCH_SIZE[@]}; do
            for tk in ${TRANSFORM_KEY[@]}; do
                for sm in ${SPLIT_MODE[@]}; do
                    sleep 0.1
                        job_file="${OUT_PATH}/jobs/run_${MODEL_CLASS}_${DATA_SET}_${OPTIMIZER}_${COND_TYPE}_${vl1}_${bs}_${nf}_${rd}_${rs}_${tk}_${N_EVAL_KEYS}_${sm}_${GS_KEY}.cmd"
                        echo "#!/bin/bash
#SBATCH -J ${MODEL_CLASS}_${DATA_SET}_${OPTIMIZER}_${COND_TYPE}_${vl1}_${bs}_${nf}_${rd}_${rs}_${tk}_${ne}_${sm}_${GS_KEY}
#SBATCH -o ${OUT_PATH}/jobs/run_${MODEL_CLASS}_${DATA_SET}_${OPTIMIZER}_${COND_TYPE}_${vl1}_${bs}_${nf}_${rd}_${rs}_${tk}_${N_EVAL_KEYS}_${sm}_${GS_KEY}.out
#SBATCH -e ${OUT_PATH}/jobs/run_${MODEL_CLASS}_${DATA_SET}_${OPTIMIZER}_${COND_TYPE}_${vl1}_${bs}_${nf}_${rd}_${rs}_${tk}_${N_EVAL_KEYS}_${sm}_${GS_KEY}.err
#SBATCH -p ${PARTITION}
#SBATCH --qos=gpu
#SBATCH -t 2-00:00:00
#SBATCH -c 4
#SBATCH --mem=50G
#SBATCH --gres=gpu:1
#SBATCH --nodelist=supergpu02pxe
#SBATCH --nice=10000
source ~/.bash_profile
conda activate tissuegpu
python3 ${CODE_PATH}/tissue/tissue/train/train_script_generative.py ${DATA_SET} ${OPTIMIZER} ${COND_TYPE} ${DOMAIN_TYPE} ${LR_KEYS} 1 1 1 1 1 1 ${vl1} ${L2_KEYS} ${USE_TYPE_COND} 1 0 1 1 1 1 1 ${bs} ${nf} ${rd} ${rs} ${tk} ${N_EVAL_KEYS} RGAN 1 1 1 1 ${BEST_NSV_PATH} ${MODEL_CLASS} ${GS_KEY} ${DATA_PATH_BASE} ${OUT_PATH} ${sm}
" > ${job_file}
                    sbatch $job_file
                done
            done
        done
    done
done