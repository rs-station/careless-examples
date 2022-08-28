import reciprocalspaceship as rs
from argparse import ArgumentParser

parser = ArgumentParser("Combine two half anomalous data sets into a single mtz")
parser.add_argument("plus_mtz")
parser.add_argument("minus_mtz")
parser.add_argument("out_mtz")
parser = parser.parse_args()

plus = rs.read_mtz(parser.plus_mtz)
minus = rs.read_mtz(parser.minus_mtz)
anom_keys = ['F(+)', 'SigF(+)', 'F(-)', 'SigF(-)', 'N(+)', 'N(-)']

out = rs.concat([
    plus,
    minus.apply_symop("-x,-y,-z"),
]).unstack_anomalous()[anom_keys]

out.write_mtz(parser.out_mtz)

