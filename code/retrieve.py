# %% --------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
from PyPDF2 import PdfReader

# %% --------------------------------------------------------------------------
reader = PdfReader(r'../biology textbook.pdf')
# -----------------------------------------------------------------------------

# %%
#find average page length of non contents page pages
page_lengths = [len(reader.pages[i].extract_text().split()) for i in [25,26,27,28,51,54,61,62,63,64,65,66,67,68,69]]
page_avg_len = sum(page_lengths)/len(page_lengths)
print(page_avg_len)

# %%
from datasets import load_dataset, Dataset
dataset = load_dataset("pubmed_qa", "pqa_labeled")  
# this is the 1000 'expertly labelled' dataset (can choose from "pqa_artificial", "pqa_labeled", "pqa_unlabeled")
data = dataset["train"].to_pandas()[["question", "context", "long_answer"]]

# %%
pub_lengths = [len("".join(data.iloc[i,1]['contexts']).split()) for i in [1,2,3,6,9,10,100,123,54,75,87,95]]
pub_avg_len = sum(pub_lengths)/len(pub_lengths)
print(pub_avg_len)

# %% --------------------------------------------------------------------------
print(f"so would need to concatenate up to {page_avg_len / pub_avg_len} points")
# -----------------------------------------------------------------------------
# %%
reader.pages[30].extract_text()
# %%
