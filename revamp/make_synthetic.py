
# %% --------------------------------------------------------------------------
import pandas as pd
from openai import OpenAI
import os
import time
import openai
# ----------------------------------------------------------------------------
df = pd.read_json(r'data/pages.json')
df.reset_index(inplace=True)
df.drop(columns='index',inplace=True)

# %% --------------------------------------------------------------------------
def gpt_enrich_data(data_inp, start_row=0, end_row=None):
    #end_row = len(data_inp) if end_row is None else end_row
    data = data_inp
    # create prompt for GPT 3.5:
    data['prompt'] = data['pages'].apply(lambda x: f"""It is your task to split text up into subsections and add informative titles for each subsection.
Each subsection must be in paragraph form and no information should be missing from the original text.
Subsections should be more than single sentences where possible.
Mark each title you create by adding the symbols "@@@" before each title.
An example subsection format is "@@@title  \n content", where you should add the subsection title and content.
This is the text:
### {x} ###""")
    data_to_enrich = data.iloc[
        start_row:end_row
    ][['prompt']]  # i dont pass all due to open AI limits

    os.environ["OPENAI_API_KEY"] = "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"
    key = "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"
    openai.api_key = key
    client = OpenAI()
    subs = []
    try:
        for i in range(0, len(data_to_enrich)):
            print(i)
            subs.append(
                client.completions.create(
                    model="gpt-3.5-turbo-instruct",
                    prompt=[data_to_enrich.iloc[i,0]],
                    #stream = True,
                    max_tokens = 2050
                )
                .choices[0].text
                #.message.content
            )
            print(i)
            time.sleep(62) if (i + 1) % 3 == 0 else None
    except:
        nout = None
    try:
        enriched_data = data_inp
        enriched_data.loc[start_row : end_row - 1, "gpt_answer"] = subs
        return enriched_data
    except:
        return subs
# -----------------------------------------------------------------------------
# %%
data = gpt_enrich_data(df)
data
# %%
try:
    data.to_csv(r"data/synthetic_data.csv")
except:
    pd.DataFrame({'responses':data}).to_csv(r"data/synthetic_data.csv")


# %% --------------------------------------------------------------------------

# combine with pages
pages = df[['pages']]
outputs = pd.read_csv(r'data/synthetic_data.csv')
outputs.drop(columns='Unnamed: 0',inplace=True)
ready_data = pages.join(outputs)
#ready_data.to_csv(r'data/ready_data.csv')
ready_data.to_json(r'data/ready_data.json')
# %%
pd.read_json(r'data/ready_data.json')

# %%
