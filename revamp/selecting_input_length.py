
# %% --------------------------------------------------------------------------
import pandas as pd
from openai import OpenAI
import os
import time
import openai
import requests

# ----------------------------------------------------------------------------
df = pd.read_json(r'data/revamped_data.json')
df.reset_index(inplace=True)
df.drop(columns='index',inplace=True)
df
# %% --------------------------------------------------------------------------

# %%
from transformers import AutoTokenizer
model_id = "TheBloke/Mistral-7B-v0.1-GPTQ"

tokenizer = AutoTokenizer.from_pretrained(
    model_id, use_fast=False, add_eos_token=True
)
# %%
for i,j in enumerate(df['pages']): 
    if len(tokenizer(j)['input_ids']) >=100 and len(tokenizer(j)['input_ids'])<900:
        print(i)
# %%
good_lens = []
bad_lens = []
for i in range(0,72):
    if len(tokenizer(df['pages'][i])['input_ids']) <= (len(tokenizer(pd.read_csv('synthetic_data_newest.csv')['responses'][i])['input_ids']))/0.7:
        good_lens.append(len(tokenizer(df['pages'][i])['input_ids']))
    else:
        bad_lens.append(len(tokenizer(df['pages'][i])['input_ids']))

# %% --------------------------------------------------------------------------
all_lens = []
for i in range(0,72):
    all_lens.append(len(tokenizer(df['pages'][i])['input_ids']))

# -----------------------------------------------------------------------------

# %%
import matplotlib.pyplot as plt
plt.hist(all_lens, bins=10, color='blue', edgecolor='black')
plt.show()

# %%
plt.hist(good_lens, bins=10, color='blue', edgecolor='black')
plt.show()
# %%
plt.hist(bad_lens, bins=5, color='blue', edgecolor='black')
plt.show()

# %% --------------------------------------------------------------------------
m = """2Topic IA — Biological MoleculesCarbohydratesEven though there is, and has been, a huge variety of different organisms on Earth, they all share some biochemistry— for example, they all contain a few carbon-based compounds that interact in similar ways.Most Carbohydrates are Polymers1) Most carbohydrates (as well as proteins and nucleic acids) are polymers.2) Polymers are large, complex molecules composed of long chains of \'=4monomers joined together.3) Monomers are small, basic molecular units.4) Examples of monomers include monosaccharides, amino acids and nucleotides.monomer e.g. monosaccharide, amino acidhrpolymer e.g. carbohydrate, proteinCarbohydrates are Made from Monosaccharides1) All carbohydrates contain the elements C, H and O.2) The monomers that they\'re made from are monosaccharides, e.g. glucose, fructose and galactose.1) Glucose is a hexose sugar — a monosaccharide with six carbon atoms in each molecule.2) There are two types of glucose, alpha (a) and beta (|3)— they\'re isomers (molecules with the same molecular formula as each other, but with the atoms connected in a different way).3) You need to know the structures of both types of glucose for your exam— it\'s pretty easy because there\'s only one difference between the two:a-glucose moleculeP-glucose moleculeCH,OH i 2H\\/ h \\ r \\Hn/ \\ ? H v k IHO c------c \\ OH /i, i \\ /H OH V^The two types of glucose have these groups reversedCH2OH\\A ~ \\rho/ VhI IH OHOH\nCondensation Reactions Join Monosaccharides Together1) A condensation reaction is when two molecules join together with the formation of a new chemical bond,and a water molecule is released when the bond is formed.2) Monosaccharides are joined together by condensation reactions.3) A glycosidic bond forms between the two monosaccharides as a molecule of water is released.4) A disaccharide is formed when two monosaccharides join together.ExampleTwo a-glucose molecules are joined together by a glycosidic bond to form maltose.HHOO. /Ha-glucoseo;h HO H O is removeda-glucoseHOHHHO5) Sucrose is a disaccharide formed from a condensation reaction between a glucose molecule and a fructose molecule.6) Lactose is another disaccharide formed from a glucose molecule and a galactose molecule.glycosidic bondO /H+ H,0lQ\'1maltoseOH1 Ml M M I I I I I n I I I I I^ If you\'re asked to show a tZ condensation reaction, don\'t ~ -- forget to put the water Iz molecule in as a product. r11111 1 1 1 n 1 1 n 1 / 11 ii 1 1Topic 1A — Biological Molecules\n3CarbohydratesHydrolysis Reactions Break Polymers Apart1) Polymers can be broken down into monomers by hydrolysis reactions.2) A hydrolysis reaction breaks the chemical bond between monomers using a water molecule. It\'s basically the opposite of a condensation reaction.3) For example, carbohydrates can be broken down into their constituent monosaccharides by hydrolysis reactions.PolymerAHydrolysis — the bond is broken by theaddition of a water molecule-OH HO--OHEven hydrolysis couldn\'t break this bond.Use the Benedict’s Test for SugarsSugar is a general term for monosaccharides and disaccharides. All sugars can be classified as reducing or non-reducing. The Benedict\'s test tests for sugars — it differs depending on the type of sugar you are testing for.1)2)3)Reducing sugars include all monosaccharides (e.g. glucose) and some disaccharides (e.g. maltose"""
len(tokenizer(m)['input_ids'])
# -----------------------------------------------------------------------------

# %%
