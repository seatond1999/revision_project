# %% --------------------------------------------------------------------------
###### starting using poroper evaluation function ###################
import pandas as pd
from peft import PeftModel
from transformers import (
    AutoTokenizer,
    GenerationConfig,
    GPTQConfig,
    AutoModelForCausalLM,
)
import torch
from openai import OpenAI
import openai
import os
import time
from PyPDF2 import PdfReader

import tensorboard

def load_model(lora_adapters, base_model):
    base_path = base_model  # input: base model
    adapter_path = lora_adapters  # input: adapters

    base_model = AutoModelForCausalLM.from_pretrained(
        base_path,
        return_dict=True,
        device_map="auto",
    )

    tokenizer = AutoTokenizer.from_pretrained(base_path)

    tokenizer.pad_token = "<unk>"
    tokenizer.padding_side = "right"
    base_model.resize_token_embeddings(len(tokenizer))
    base_model.config.eos_token_id = tokenizer.eos_token_id
    base_model.config.pad_token_id = tokenizer.pad_token_id

    # Load LoRA adapter and merge
    return tokenizer, PeftModel.from_pretrained(base_model, adapter_path)

class retriever:

    def __init__(self,book,model,tokenizer):
        self.book = book
        self.model = model
        self.tokenizer = tokenizer
        self.page_lag = None

    def get_contents(self):
        prompt = """<s>[INST] @@@ Instructions:
It is your task to classify whether a string corresponds to the contents page of a pdf book.
A contents page includes chapter titles and page numbers.
The first word of you answer must be "Yes" or "No"
You must reply "yes" if the string is from the contents page, and "no" if it is not the contents page.

@@@ Example:
If this is the string: ### 'Contents \nIntroduction  v\nUnit 1 Business and its environment 2\n1: Enterprise  3\n2: Business structure 15\n3: Size of business 29\n4: Business objectives 38\n5: Stakeholders in a business 51\n6: Business structure (A Level) 61\n7: Size of business (A Level) 70\n8:  External in fl uences on business activity (A Level) 76\n9:  External economic in fl uences on business behaviour (A Level) 98\nUnit 2 People in organisations 124\n10: Management and leadership 125\n11: Motivation 137\n12: Human resource management 159\n13:  Further human resource management (A Level) 170\n14: Organisation structure (A Level) 187\n15: Business communication (A Level) 200\nUnit 3 Marketing 212\n16: What is marketing? 213\n17: Market research 231\n18: Th e marketing mix – product and price 252\n19:  Th e marketing mix – promotion and place 273\n20: Marketing planning (A Level) 297\n21: Globalisation and international marketing (A Level) 318iiiContents ###

The correct answer is: Yes

@@@ Example:
If this is the string: ### 'vi ABOUT THIS BOOK\nABOUT THIS BOOK\nThis book is written for students following the Pearson Edexcel International Advanced Subsidiary (IAS) \nBiology specification. This book covers the full IAS course and the first year of the International A Level  (IAL) course.\nThe book contains full coverage of IAS units (or exam papers) 1 and 2. Each unit in the specification has  \ntwo topic areas. The topics in this book, and their contents, fully match the specification. You can refer to the Assessment Overview on page x for further information. Students can prepare for the written Practical Paper (unit 3) by using the IAL Biology Lab Book (see page viii of this book).\nEach topic is divided into chapters and sections to break the content down into manageable chunks.  \nEach section features a mix of learning and activities. \nLearning objectives\nEach chapter starts with a listof key assessment objectives.\nDid you know?Interesting facts help you remember the key concepts.CheckpointQuestions at the end of each section check understanding of the key learning points in each chapter.Subject vocabularyKey terms are highlighted in blue in the text. Clear definitions are provided at the end of each section for easy reference, and are also collated in a glossary at the back of the book. Worked examples show you how to work through questions, and set out calculations.Specification referenceThe exact specification references covered in the section are provided. Exam hintsTips on how to answer exam-style questions and guidance for exam preparation. Orange Learning Tips help you focus your learning and avoid common errors. \nUncorrected proof, all content subject to change at publisher discretion. Not for resale, circulation or distribution in whole or in part. ©Pearson 2018' ###

The correct answer is: No

@@@ Question:
This is the string: ### """
        for i in range(0,30):
            input_ids = self.tokenizer(prompt+book.pages[i].extract_text()+' ### [/INST]',return_tensors='pt').input_ids.cuda()
            output = self.tokenizer.decode((self.model.generate(inputs=input_ids, temperature=0.7, do_sample=True, top_p=0.95, top_k=40, max_new_tokens=400000))[0])
            


            

base_model_path = "TheBloke/Mistral-7B-Instruct-v0.1-GPTQ"
book = PdfReader(r"../../edexcel_a_level_physics_student_book_1.pdf")
retrieve = retriever(book,1,2)


# %% --------------------------------------------------------------------------
retrieve.get_contents()
# -----------------------------------------------------------------------------


# %%
book.pages[1].extract_text()