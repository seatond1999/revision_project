# %% --------------------------------------------------------------------------
from PyPDF2 import PdfReader
import pandas as pd

# -----------------------------------------------------------------------------

# %% --------------------------------------------------------------------------
# aqa biology
pages = [
    8,
    9,
    10,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    40,
    41,
    42,
    43,
    44,
    45,
    46,
    48,
    50,
    52,
    54,
    58,
    60,
    62,
    64,
    66,
    70,
    72,
    74,
    76,
    78,
    80,
    81,
    82,
    84,
    86,
    88,
    90,
    92,
    94,
    95,
    98,
    100,
    102,
    104,
    108,
    110,
    150,
    152,
    154,
    156,
    158,
    180,
    182,
    184,
    186,
    188,
    202,
    203,
    205,
    207,
    208,
    210,
]
pages = list(map(lambda x: x - 1, pages))
reader = PdfReader(r"../../CGP-AQA-Biology-A-Level-ariadanesh.com_.pdf")
aqabio_data = [reader.pages[i].extract_text() for i in pages]
# -----------------------------------------------------------------------------

# %% --------------------------------------------------------------------------
# cambridge business book
pages = [
    15,
    16,
    17,
    25,
    26,
    27,
    28,
    37,
    38,
    39,
    50,
    51,
    53,
    54,
    55,
    210,
    222,
    443,
    444,
    445,
    456,
    559,
    557,
    552,
]
pages = list(map(lambda x: x - 1, pages))
reader = PdfReader(r"../../cambridge business textbook.pdf")
biscam_data = [reader.pages[i].extract_text() for i in pages]
# %%
# edexcel biology pearson
pages = [15, 18,21,22,23,24,25,26,27,28,29,55,56,62,63,64,65,66,67,75,77,78,79]
pages = list(map(lambda x: x - 1, pages))
reader = PdfReader(r"../../biology textbook.pdf")
biotext_data = [reader.pages[i].extract_text() for i in pages]

# %%
data = aqabio_data + biscam_data + biotext_data
# %%
data_df = pd.DataFrame({'pages':data})
data_df.to_csv(r'/pages.csv')

# %%
pd.read_csv(r'\data\pages.csv')

# %%
hi = pd.read_csv(r"../enriched_data.csv")

# %%
hi.to_csv(r'/ready_data.csv')