mkdir merge #First make an output directory
careless mono \
  --anomalous \
  --merge-half-datasets \
  --mlp-layers=10 \
  --image-layers=2 \
  --dmin=1.8 \
  --iterations=30_000 \
  "dHKL,xobs,yobs" \
  unmerged.mtz \
  merge/thermolysin


mkdir merge_eo #eo for Ewald offset
careless mono \
  --anomalous \
  --merge-half-datasets \
  --mlp-layers=10 \
  --image-layers=2 \
  --dmin=1.8 \
  --iterations=30_000 \
  "dHKL,xobs,yobs,ewald_offset" \
  unmerged.mtz \
  merge_eo/thermolysin

