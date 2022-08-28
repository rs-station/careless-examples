import reciprocalspaceship as rs
from argparse import ArgumentParser

parser = ArgumentParser("Split an mtz into anomalous half-datasets")
parser.add_argument("unmerged_mtz")
parser.add_argument("-p", help="Plus Friedel mates mtz. Default 'friedel_plus.mtz'", default="friedel_plus.mtz", type=str)
parser.add_argument("-m", help="Minus Friedel mates mtz. Default 'friedel_minus.mtz'", default="friedel_minus.mtz", type=str)
parser = parser.parse_args()

ds = rs.read_mtz(parser.unmerged_mtz)

plus = (ds.hkl_to_asu()["M/ISYM"].to_numpy() % 2 == 1)

#Include all centrics in friedel plus
centrics = ds.label_centrics().CENTRIC.to_numpy()
plus |= centrics

ds[plus].copy().write_mtz(parser.p)
ds[~plus].copy().write_mtz(parser.m)


