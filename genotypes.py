from    collections import namedtuple

Genotype = namedtuple('Genotype', 'gene weight')

PRIMITIVES_ops1 = [
    'skip',
    'normalization_channel',
    'normalization_global',
    'normalization_sep',

]

PRIMITIVES_ops2 = [

    'multiplication',
    'sub_abs',
    'sub_squ',
    'cov',

]


