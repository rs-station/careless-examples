#!/bin/bash
#SBATCH -J refine
#SBATCH -p shared,serial_requeue,seas_compute  # partition (queue)
#SBATCH -n 4         # 8 cores
#SBATCH --mem 32G    # memory pool for all cores
#SBATCH -t 0-00:15   # time (D-HH:MM)

# source your copy of phenix here!
# source ../../../../../../phenix-1.20.1-4487/phenix_env.sh
source /n/hekstra_lab_tier0/Lab/garden/phenix_1_20/phenix-1.20.1-4487/phenix_env.sh

REF="../../../refinement"
# Setup run directory
mkdir run_omit_1
cd run_omit_1

# Copy refine.eff
cp ${REF}/refine_omit.eff .

# Run refinement
phenix.refine refine_omit.eff --overwrite
