# %% --------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
from datasets import load_dataset, Dataset
from openai import OpenAI
import os
import time

# %% --------------------------------------------------------------------------
# Set Random State
# -----------------------------------------------------------------------------
rng = np.random.RandomState(123)


# %% --------------------------------------------------------------------------
def load_data():
    dataset = load_dataset(
        "pubmed_qa", "pqa_labeled"
    )  # this is the 1000 'expertly labelled' dataset (can choose from "pqa_artificial", "pqa_labeled", "pqa_unlabeled")
    data = dataset["train"].to_pandas()[["question", "context", "long_answer"]]
    data["context"] = data["context"].apply(lambda x: "".join(x["contexts"]))
    extra_data = data.iloc[664:]
    data = data.iloc[0:664, :]  # as dont have many credits on
    return data, extra_data


data = load_data()
# -----------------------------------------------------------------------------

# %%
enriched_data = pd.read_csv(r"../enriched_data.csv")
enriched_data.drop(columns=["Unnamed: 0"], inplace=True)
# enriched_data = data[0]
# enriched_data.loc[0:664, "gpt_question"] = "test_value"
extra_data = data[1]


# %%
def create_multiq_data(data_unprepared, single_qs):
    # prep single question data
    single_qs = single_qs[["context", "question"]]
    single_qs.rename(
        columns={"context": "context", "question": "combined_questions"}, inplace=True
    )
    single_qs["context_length"] = single_qs.apply(
        lambda x: len(x["context"].split()), axis=1
    )
    single_qs = single_qs.sort_values(by="context_length", ascending=True)
    single_qs.reset_index(inplace=True)
    single_qs = single_qs.head(166)
    ##
    df = data_unprepared
    df["combined_questions"] = df["question"] + "\n" + df["gpt_question"]
    df["context_length"] = df.apply(lambda x: len(x["context"].split()), axis=1)
    df = df.sort_values(by="context_length", ascending=False)
    df.reset_index(inplace=True)
    # create quad Qs with longest contexts:
    first_quart = df.loc[0:165, ["context", "combined_questions"]]
    first_quart.reset_index(inplace=True)
    second_quart = df.loc[166:331, ["context", "combined_questions"]][
        ::-1
    ]  # can tidy later
    second_quart.reset_index(inplace=True)
    quad_qs = pd.concat([first_quart, second_quart], axis=1)
    quad_qs["full_context"] = quad_qs.iloc[:, 1] + "\n" + quad_qs.iloc[:, 4]
    quad_qs["full_questions"] = quad_qs.iloc[:, 2] + "\n" + quad_qs.iloc[:, 5]
    # create triple Qs:
    third_quart = df.loc[332:497, ["context", "combined_questions"]]
    third_quart.reset_index(inplace=True)
    triple_qs = pd.concat([third_quart, single_qs], axis=1)
    triple_qs["full_context"] = triple_qs.iloc[:, 1] + "\n" + triple_qs.iloc[:, 4]
    triple_qs["full_questions"] = triple_qs.iloc[:, 2] + "\n" + triple_qs.iloc[:, 5]
    # create double Qs
    forth_quart = df.loc[498:663, ["context", "combined_questions"]]
    forth_quart.reset_index(inplace=True)
    double_qs = forth_quart
    double_qs.rename(
        columns={"context": "full_context", "combined_questions": "full_questions"},
        inplace=True,
    )
    # put together!
    combined = pd.concat(
        [
            quad_qs[["full_context", "full_questions"]],
            triple_qs[["full_context", "full_questions"]],
            double_qs[["full_context", "full_questions"]],
        ]
    )
    combined["context_length"] = combined.apply(
        lambda x: len(x["full_context"].split()), axis=1
    )

    return combined


d = create_multiq_data(enriched_data, extra_data)
d

# %% --------------------------------------------------------------------------
# context length dist per group
import matplotlib.pyplot as plt

print(plt.hist(d.iloc[0:166, 2]))
print(plt.hist(d.iloc[166:332, 2]))
print(plt.hist(d.iloc[332:498, 2]))

# -----------------------------------------------------------------------------


# %%
d.to_csv("lemmesee.csv")
# %%
a = pd.DataFrame({"one": [1, 2, 3, 4], "three": [1, 2, 3, 4]})
b = pd.DataFrame({"one": [1, 2, 3, 4], "four": [1, 2, 3, 4]})
pd.concat([a, b], axis=1)
# %%
first_quart = enriched_data.loc[0:165, ["context", "combined_questions"]]
second_quart = enriched_data.loc[166:331, ["context", "combined_questions"]]
pd.concat([first_quart, second_quart], axis=1, ignore_index=True)
# %%
pd.concat([first_quart, second_quart], axis=1)
# %%
second_quart.reset_index(inplace=True)


# %% --------------------------------------------------------------------------
# 
# -----------------------------------------------------------------------------
pd.read_csv(r'../ready_data.csv').loc[1,'full_questions']
# %%

# %%
