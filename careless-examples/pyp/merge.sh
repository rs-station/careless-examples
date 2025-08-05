out=merge
mtz_0=off.mtz
mtz_1=2ms.mtz
mkdir $out

careless poly \
  --separate-files \
  --disable-image-scales \
  --iterations=30_000 \
  --merge-half-datasets \
  --wavelength-key='Wavelength' \
  "X,Y,Wavelength,dHKL,Hobs,Kobs,Lobs" \
  $mtz_0 \
  $mtz_1 \
  $out/pyp


