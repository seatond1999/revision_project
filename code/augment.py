# %% --------------------------------------------------------------------------
import pandas as pd
from datasets import load_dataset, Dataset
from openai import OpenAI
import os
import time
import matplotlib.pyplot as plt
import openai


# -----------------------------------------------------------------------------
# %% -------------------------------------------------------------------------# -----------------------------------------------------------------------------
def load_data():
    dataset = load_dataset(
        "pubmed_qa", "pqa_labeled"
    )  # this is the 1000 'expertly labelled' dataset (can choose from "pqa_artificial", "pqa_labeled", "pqa_unlabeled")
    data = dataset["train"].to_pandas()[["question", "context", "long_answer"]]
    data["context"] = data["context"].apply(lambda x: "".join(x["contexts"]))
    extra_data = data.iloc[664:]
    data = data.iloc[0:664, :]  # as dont have many credits on
    return data, extra_data


if __name__ == "__main__":
    data = load_data()


# %% --------------------------------------------------------------------------
# function to enrivching data - add line which puts it on to table and joins with other question
def gpt_enrich_data(data_inp, start_row=0, end_row=None):
    end_row = len(data_inp) if end_row is None else end_row
    # create prompt for GPT 3.5:
    data_to_enrich = pd.DataFrame(
        {
            "prompt": data_inp.apply(
                lambda x: """You are an examiner who will be provided with context and an example question based on the context which will be separated by the delimiter '###'. You must ask a new question which can be answered using only the context provided.\nContext: ### """
                + x["context"]
                + " ###\nExample question based on the context: ### "
                + x["question"]
                + " ###\nQuestion based on the context: ### ",
                axis=1,
            ),
        }
    )
    # pass each prompt into GPT and put return in new column:
    data_to_enrich = data_to_enrich.iloc[
        start_row:end_row
    ]  # i dont pass all due to open AI limits

    os.environ["OPENAI_API_KEY"] = "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"
    key = "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"
    openai.api_key = key
    client = OpenAI()
    new_questions = []
    try:
        for i in range(0, len(data_to_enrich)):
            print(i)
            new_questions.append(
                client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": data_to_enrich.iloc[i, 0]}],
                )
                .choices[0]
                .message.content
            )
            print(i)
            time.sleep(62) if (i + 1) % 3 == 0 else None
    except:
        pass
    try:
        enriched_data = data_inp
        enriched_data.loc[start_row : end_row - 1, "gpt_question"] = new_questions
        return enriched_data
    except:
        return new_questions


if __name__ == "__main__":
    enriched_data = gpt_enrich_data(data[1], 336,None)
# if doing in batches of 200 would do this:
# enriched_data = gpt_enrich_data(enriched_data, 200,401) etc spaced 1 day apart lol

# enriched_data.to_csv(r'../enriched_data.csv')

# %% --------------------------------------------------------------------------
## add mulitQ
try:
    enriched_data.to_csv("from_336.csv")
except:
    pd.DataFrame(enriched_data).to_csv('from_336.csv')
# -----------------------------------------------------------------------------

# %% --------------------------------------------------------------------------
some_more = enriched_data
# -----------------------------------------------------------------------------


# %% --------------------------------------------------------------------------
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
    combined.reset_index(inplace=True)
    combined.drop(columns=["index"], inplace=True)
    return combined


if __name__ == "__main__":
    enriched_data = pd.read_csv(r"../enriched_data.csv")
    enriched_data.drop(columns=["Unnamed: 0"], inplace=True)
    extra_data = data[1]

    ready_data = create_multiq_data(enriched_data, extra_data)
# -----------------------------------------------------------------------------

# %% --------------------------------------------------------------------------


# context length dist per group
def context_length_dist():
    print(plt.hist(ready_data.iloc[0:166, 2]))
    print(plt.hist(ready_data.iloc[166:332, 2]))
    print(plt.hist(ready_data.iloc[332:498, 2]))


if __name__ == "__main__":
    context_length_dist()
# -----------------------------------------------------------------------------

# %% --------------------------------------------------------------------------
if __name__ == "__main__":
    ready_data.to_csv(r"../ready_data.csv")
# -----------------------------------------------------------------------------
