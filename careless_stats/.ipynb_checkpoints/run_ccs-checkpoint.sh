#!/bin/bash

#please activate your desired careless environment here!
eval "$(conda shell.bash hook)"
conda activate careless

dir_header=$1
for dir in ./careless_runs/${dir_header}
do
  echo $dir
  cd $dir
  echo "$(pwd)"
  IFS='_' read -ra ADDR <<< "$dir"
  TMP_R=${ADDR[-1]}
  echo ${TMP_R:0:-1}
  careless.ccpred thl_1p8A_grid_predictions_[0-1].mtz -o ccpred_res.csv --image ccpred_careless.png -b 10 -m spearman 
  careless.ccanom *xval_both.mtz -o ccanom_res.csv --image ccanom_res.png -b 10 # -m spearman
  careless.ccanom *xval_both.mtz -o ccanom_overall.csv -b 1 # -m spearman
  python ../../scripts/custom_ccanom.py *xval_both.mtz -o ccanom_overall.csv -b 1 -c 2 --image ccanom_overall.png
  careless.cchalf *xval_both.mtz -o cchalf_res.csv --image cchalf_res.png -b 10 -m spearman
  careless.cchalf *xval_both.mtz -o cchalf_overall.csv -b 1 -m spearman
  
  mtz_filename=run_omit_1/2tli_occ0_ions_refine_001.mtz
  pdb_filename=run_refinement/2tli_refine_001.pdb
  output_csv=peak_heights

  python ../../scripts/anomalous_peak_heights.py ${mtz_filename} ${pdb_filename} [Zn,Ca] ${output_csv}.csv
  python ../../scripts/anomalous_peak_heights.py ${mtz_filename} ${pdb_filename} [Zn,Ca] ${output_csv}_W_dh.csv W_dh
  python ../../scripts/anomalous_peak_heights.py ${mtz_filename} ${pdb_filename} [Zn,Ca] ${output_csv}_W_dh_d.csv W_dh_d
  python ../../scripts/anomalous_peak_heights.py ${mtz_filename} ${pdb_filename} [Zn,Ca] ${output_csv}_W_dh_e.csv W_dh_e
   cd ../..
done
