#!/bin/bash
#SBATCH --job-name=thl_dw
#SBATCH -p gpu_requeue,seas_gpu # partition (queue)
#SBATCH --mem 32G # memory pool for all cores
#SBATCH -t 0-01:20 # time (D-HH:MM)
#SBATCH --gres=gpu:1
#SBATCH --constraint=v100
#SBATCH --array=2-16
#SBATCH -o myoutput_%j.out
#SBATCH -e myoutput_%j.err

#  ** PROCESS GRID PARAMETERS **
PARAM_FILE=slurm_params.txt
#IL	MLPL    ITER    STDOF   PEF     rDW	RU
MY_PARAMS=$(sed "${SLURM_ARRAY_TASK_ID}q;d" ${PARAM_FILE})
TMP=(${MY_PARAMS///})
IL="${TMP[0]}"
MLPL="${TMP[1]}"
ITER="${TMP[2]}"
STDOF="${TMP[3]}"
PEF="${TMP[4]}"
R="${TMP[5]}"
RU="${TMP[6]}"

MODE="mono"
DMIN=1.8
TEST_FRACTION=0.1
SEED=$RANDOM
BASENAME=thl_1p8A_grid
HALF_REPEATS=3


INPUT_MTZS=(
    ../unmerged_mtzs/friedel_plus.mtz
    ../unmerged_mtzs/friedel_minus.mtz
)

DW_LIST=None,0
USE_DW="DW"
# WILSON r ARRAY (if n+1 items, call sbatch with --array=0-n)
# r_string="scale=5; 1-0.4^${SLURM_ARRAY_TASK_ID}"
# R=$(bc -l <<< ${r_string} )
# RX=$R
DWR_LIST=0.,${R}


#Source your installation of careless here. To install careless, see: https://github.com/rs-station/careless
eval "$(conda shell.bash hook)"
conda activate careless


OUT=merge_${SLURM_JOB_ID}_${SEED}_${MODE}_cl3_mc1_grid_${SLURM_ARRAY_TASK_ID}
mkdir -p $OUT
cp $0 $OUT
cat $0 > $OUT/slurm_script


SECONDS=0
CARELESS_ARGS=(
    --mc-samples=1
    --learning-rate=0.001
    --separate-files
    --merge-half-datasets
    --half-dataset-repeats=$HALF_REPEATS
    --mlp-layers=$MLPL
    --image-layers=$IL
    --dmin=$DMIN
    --iterations=$ITER
#    --positional-encoding-frequencies=$PEF
#    --positional-encoding-keys="xobs,yobs"
#    --wilson-prior-b 18.0
    --test-fraction=$TEST_FRACTION
    --seed=$SEED
#    --double-wilson-parents=$DW_LIST #see below
#    --double-wilson-r=$DWR_LIST
    --mlp-width=4
)
# WARNING: bash [] comparisons only work for integers
c=$(echo "$R > -0.01" | bc)
if [ $c = '1' ]
then
  CARELESS_ARGS+=(--double-wilson-parents=${DW_LIST})
  CARELESS_ARGS+=(--double-wilson-r=${DWR_LIST})
fi
if [ $IL -lt 0 ]
then
  CARELESS_ARGS+=( --disable-image-scales)
fi
if [ $RU -gt 0 ]
then
  CARELESS_ARGS+=( --refine-uncertainties)
fi
if [ $STDOF -gt 0 ]
then
  CARELESS_ARGS+=( --studentt-likelihood-dof=$STDOF)
fi
CARELESS_ARGS+=("dHKL,xobs,yobs,ewald_offset")

echo $CARELESS_ARGS
echo "Careless version (from sbatch script): ${CARELESS_VERSION}"
echo "Input MTZs: ${INPUT_MTZS[@]}" > ./$OUT/inputs_params.log
echo "Args: $MODE ${CARELESS_ARGS[@]}" >> ./$OUT/inputs_params.log
conda list > ./$OUT/conda_env_record.log

careless $MODE ${CARELESS_ARGS[@]} ${INPUT_MTZS[@]} $OUT/$BASENAME


# --- Cleanup --- #
echo "careless run complete."
#touch ./$OUT/inputs_params.log
mv myoutput*${SLURM_JOB_ID}* $OUT #DH added

DURATION=$SECONDS
TITLE="Slurm: careless"
MESSAGE="Job $SLURM_JOB_ID:careless finished on $HOSTNAME in $(($DURATION / 60)) minutes."
echo $MESSAGE
