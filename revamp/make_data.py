# %% --------------------------------------------------------------------------
from PyPDF2 import PdfReader
import pandas as pd

# -----------------------------------------------------------------------------

# %% --------------------------------------------------------------------------
# aqa biology
[]
pages = [
    8,
    9,
    10,#
    12,#
    14,
    15,#
    16,
    17,#
    18,
    19,#
    20,21,#
    22,23,##
    24,25,##
    26,28,
    30,31,32,34,#
    36,38,##
    40,##
    42,##
    44,##
    46,48,##
    50,##
    52,54,##
    55,56,##
    58,##
    60,##
    62,
    64,66,68,##
    70,##
    72,##
    74,##
    76,##
    78,79,#
    81,82,#
    84,85,##
    86,##
    88,##
    90,##
    92,##
    94,95,#
    98,99,##
    100,##
    102,##
    104,##
    108,##
    110,##
    112,113,114,116,118,#
    120,121,122,124,126,128,130,132,134,136,138,140,#
    142,143,#
    145,146,148,150,152,154,156,#
    162,163,
    165,166,#
    192,193,194,#
    207,208,#
    214,215,#
    168,170,172



]
pages = list(map(lambda x: x - 1, pages))
reader = PdfReader(r"../../CGP-AQA-Biology-A-Level-ariadanesh.com_.pdf")
aqabio_data = [reader.pages[i].extract_text() for i in pages]
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------

# %% --------------------------------------------------------------------------
# cambridge business book
pages = [
    11,12,
    14, #14,15
    17,19,
    24,32,42,
    37,38,39,
    46,51,
    53,54,55,#53
    59,60,
    60,71,73,94,101,107,109,116,118,125,134,148,
]
pages = list(map(lambda x: x - 1, pages))
reader = PdfReader(r"../../cambridge business textbook.pdf")
biscam_data = [reader.pages[i].extract_text() for i in pages]




# %%
# edexcel biology pearson
pages = [
    14,15,16,17,
    18,19,
    21,23,
    24,
    26,27,30,
    38,39,
    41,42,43,44,
    45,
    47,48,49,
    51,52,
    54,55,56,
    62,63,
    66,67,
    69,73,
    75,
    79,
]
pages = list(map(lambda x: x - 1, pages))
reader = PdfReader(r"../../biology textbook.pdf")
biotext_data = [reader.pages[i].extract_text() for i in pages]

# %% --------------------------------------------------------------------------
# hodder
pages = [
    8,
    10,
    13,15,16,#13,14
    18,
]
pages = list(map(lambda x: x - 1, pages))
reader = PdfReader(r"../../AQA-8461-HODDER-SAMPLE.pdf")
hod_data = [reader.pages[i].extract_text() for i in pages]


# %% --------------------------------------------------------------------------

final = aqabio_data + (biscam_data) + (biotext_data) + (hod_data)
final = list(set(final))
final = pd.DataFrame({'pages':final})
final.to_json('final_pages.json')


# %%
from transformers import AutoTokenizer
model_id = "TheBloke/Mistral-7B-v0.1-GPTQ"

tokenizer = AutoTokenizer.from_pretrained(
    model_id, use_fast=False, add_eos_token=True
)
# %%
#hi = aqabio_data[0] +aqabio_data[1]+aqabio_data[2]+aqabio_data[3]
#hi = biotext_data[11]+biotext_data[12]+biotext_data[13]+biotext_data[14]+biotext_data[15]
lens = []
for i in final:
    lens.append(len(tokenizer(i)['input_ids']))


# %%
plz = pd.read_json('synthetic_data_newest.json')
for i in range(0,len(plz)):
    if len(tokenizer(plz['pages'][i])['input_ids']) >= len(tokenizer(plz['sums'][i])['input_ids'])/0.6:
        print(i)


# %% --------------------------------------------------------------------------
plz['sums'][141]
# -----------------------------------------------------------------------------

# %%
#for test ...
pages = [
    206,224,226,266,
    271,272
]
pages = list(map(lambda x: x - 1, pages))
reader = PdfReader(r"../../edexcel_a_level_physics_student_book_1.pdf")
physics = [reader.pages[i].extract_text() for i in pages]
e = physics[4]+physics[5]
physics=physics[0:4]
physics.append(e)
pd.DataFrame({'test_set':physics}).to_csv('test_data.csv')
# %%

