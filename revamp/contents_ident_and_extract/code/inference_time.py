# %% --------------------------------------------------------------------------
#
# -----------------------------------------------------------------------------

import pandas as pd
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GPTQConfig,
    TrainingArguments,
    TrainerCallback,
    BitsAndBytesConfig,
)
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer
import os
import tempfile
from huggingface_hub import login
from datetime import datetime as dt
import torch
import tensorboard
# -----------------------------------------------------------------------------


## timer
def timer(func):
    def do(*args):
        before = dt.now()
        perform = func(*args)
        after = dt.now()
        print(after - before)
        return perform

    return do


# %%
token = "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"
login(token=token)

def prep_data():
    data = pd.read_csv(r"../data/prepared_data.csv")
    data = data[data['extraction'] != 'IGNORE']
    data_aq = pd.DataFrame(
        {
            "content": data.apply(
                lambda x: f"""<s>[INST] @@@ Instructions:
It is your task to extract the chapters and corresponding page numbers from a string which was created from the contents page of a pdf book.
You must return a list of the chapters and page numbers.
Put each chapter and its page number on its own line, and separate chapters titles from page numbers with a "---".
You will be penalised for separating chapters with anything that is not "---"
For example the first 2 chapters of a contents page should be in the following format: "chapter 1 title --- chapter 1 page number \n chapter 2 title --- chapter 2 page number"

@@@ Question:
string which was created from the contents page of a pdf book: ### {x['contents_page']} ### [/INST]

Output: {x['extraction']}""",
                axis=1,
            )
        }
    )
    dataset = Dataset.from_pandas(data_aq)
    return data_aq,dataset


# %% --------------------------------------------------------------------------


token = "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"
login(token=token)
data = prep_data()[0]

##load model and tokenizer
model_id = "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ"

tokenizer = AutoTokenizer.from_pretrained(
    model_id, use_fast=False
)

quantization_config_loading = GPTQConfig(
    bits=4,
    disable_exllama=True, 
    tokenizer=tokenizer,
)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quantization_config_loading,
    device_map="auto",
    torch_dtype=torch.float16,
)

tokenizer.pad_token = "<unk>"
tokenizer.padding_side = "right"
model.resize_token_embeddings(len(tokenizer))
model.config.eos_token_id = tokenizer.eos_token_id
model.config.pad_token_id = tokenizer.pad_token_id




    