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


def inference(tokenizer, model, context):
    #context = context.replace('\n',' ')
    #context = context.replace('\xa0–',' ')
    #context = context.replace('\xa0',' ')
    prompt = f"""[INST]You must split text up into subsections and add informative titles for each subsection.
Each subsection must be in paragraph form and all information should be included from the original text.
You will be penalized for removing information from the original text.
Mark each title you create by adding the symbols "@@@" before each title and placing the title on its own line.
An example subsection format is "@@@title \n content", where you should add the subsection title and content.
This is the text:
### {context} ### [/INST]
Output: """

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda:0")

    generation_config = GenerationConfig(
        do_sample=True,
        top_p=0.95, top_k=40,
        temperature=0.7,
        max_new_tokens=150000,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    out = model.generate(inputs=input_ids, generation_config=generation_config)
    decoded_output = tokenizer.decode(out[0], skip_special_tokens=True)
    decoded_output = decoded_output[
            decoded_output.find("[/INST]")
            + len("[/INST]") :
        ]

    return decoded_output

# -----------------------------------------------------------------------------


# %% --------------------------------------------------------------------------
if __name__ == "__main__":
    adapter_path = "/home/seatond/revision_project/revamp/gp4_rank32_lr2.2e-05_target7_epochs2_laplha64_batch2_gradacc2"
    base_model_path = "TheBloke/Mistral-7B-Instruct-v0.1-GPTQ"
    #eval_path = r"../evaluation_data.csv"
    #context_list = list(pd.read_csv(eval_path)["contexts"])
    #tokenizer, model = load_model(adapter_path, base_model_path)
    test_data = pd.read_csv('test_data.csv').iloc[1:2]
    outputs = []
    counter = 1
    for i in test_data['test_set']:
    #for i in ['This chapter covers syllabus \nsection AS Level 1.1.\nDULIP STARTS HIS BUSINESS\nDulip lives in a large country with many natural \nresources, such as coal and timber. He plans to start \na business growing and cutting trees to sell as timber. \nHe wants to buy a forest from a farmer and cut down a \nfixed number of trees each year. As Dulip is concerned \nabout his environment country’s, he will plant two new \ntrees for each one he cuts down. He has been planning \nthis business venture for some time. He has visited a \nbank to arrange a loan. He has contacted suppliers of \nsaws and other equipment to check on prices. Dulip \nhas also visited several furniture companies to see \nif they would be interested in buying wood from the \nforest. In fact, he did a great deal of planning before he \nwas able to start his business.\nDulip is prepared to take risks and will invest his \nown savings, as well as using the bank loan, to set \nup the business. He plans to employ three workers \nto help him to start with. If the business is a success, \nthen he will also try to sell some of the timber abroad. \nHe knows that timber prices are high in some foreign \nmarkets. A ft er several months of planning, he was able \nto purchase the forest.\nPoints to think about:\n■ Why do you think Dulip decided to own and run his own \nbusiness rather than work for another firm?\n■ Why was it important to Dulip that he should do so much \nplanning before starting his business?\n■ D o  y o u  t h i n k  D u l i p  w i l l  m a k e  a  s u c c e s s f u l  e n t r e p r e n e u r ?\n■ Are new businesses such as Dulip’s good for a country’s \neconomy?1 Enterprise\nIntroducing the topic\nIntroduction\nMany business managers are paid high salaries to take risks \nand make decisions that will in fl uence the future success \nof their business. Much of this book is concerned with \nhow these decisions are made, the information needed to \nmake them and the techniques that can assist managers in \nthis important task. However, no student of Business can \nhope to make much progress in the study of this subject \nunless they have a good understanding of the economic \nenvironment in which a business operates. Business activity \ndoes not take place in isolation from what is going on \naround it. Th e very structure and health of the economy \nwill have a great impact on how successful business activity \nis. Th e central purpose of this whole unit, ‘Business and \nits environment’, is to introduce the inter-relationships \nbetween businesses, the world in which they operate and the \nlimits that governments impose on business activity. Th i s  \nfi rst chapter explains the nature of business activity and the \nrole of enterprise in setting up and developing businesses.\nThe purpose of business activity\nA business is any organisation that uses resources to \nmeet the needs of customers by providing a product or \nservice that they demand. Th ere are several stages in the \nproduction of fi nished goods. Business activity at all stages \ninvolves creating and adding value to resources, such \nas raw materials and semi- fi nished goods, and making \nthem more desirable to \xa0– and thus valued by \xa0– the fi nal \npurchaser. Without business activity, we would all still be \nentirely dependent on the goods that we could make or \ngrow ourselves \xa0– as some people in virtually undiscovered \nnative communities still are. Business activity uses the \nscarce resources of our planet to produce goods and \nservices that allow us all to enjoy a very much higher \nstandard of living than would be possible if we remained \nentirely self-su ffi  cient.On completing this chapter, you will be able to:\n■ understand what business activity involves\n■ recognise that making choices as a result of the \n‘economic problem’ always results in opportunity cost\n■ analyse the meaning and importance of creating value\n■ recognise the key characteristics of successful \nentrepreneurs■ assess the importance of enterprise and entrepreneurs \nto a country’s economy\n■ understand the meaning of social enterprise and the \ndiff erence between this and other businesses.\n3']:
        print(counter)
        outputs.append(inference(tokenizer, model, i))
        print(counter)
        counter +=1
    #outputs = inf_on_all(test_data)

# %% --------------------------------------------------------------------------

# -----------------------------------------------------------------------------

# %% --------------------------------------------------------------------------
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

model_name_or_path = "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ"
# To use a different branch, change revision
# For example: revision="gptq-4bit-32g-actorder_True"
model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                             device_map="auto",
                                             trust_remote_code=False,
                                             revision="main")

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
# -----------------------------------------------------------------------------

# %% --------------------------------------------------------------------------
from auto_gptq import exllama_set_max_input_length
model = exllama_set_max_input_length(model, max_input_length=3609)
prompt = """Please return the following text back to me but split into main subsections which you identify, where each subsection is in paragraph form and no information is missing from the original text. Also add a title for the subsection at the start of that subsection and label it with the indicator "@@@". An example subsection format is "@@@SUBSECTION TITLE  \n SUBSECTION CONTENT". This is the text: ### 'So cash fl ow is certainly important \xa0– especially to small \nbusiness start-ups. Cash- fl ow planning is vital for \nentrepreneurs because:\n■ new business start-ups are o ft en o ff ered much less time to \npay suppliers than larger, well-established firms \xa0– they are \ngiven shorter credit periods\n■ banks and other lenders may not believe the promises of \nnew business owners as they have no trading record, they \nwill expect payment at the agreed time\n■ finance is o ft en very tight at start-up, so not planning \naccurately is of even more significance for new businesses.\nClearly, the consequences of not having a suitable level of \ncash in the business can be very serious \xa0– or even life-\nthreatening for that business.\nCash and profit \xa0– what’s the di ff e r e n c e ?\nFor many failed business owners there was no di ff erence \nbetween cash and pro fi t\xa0– which is why their business \ncollapsed. All successful entrepreneurs and business \nmanagers, in contrast, understand that these two fi nancial \nconcepts do not have the same meaning or signi fi cance for a \nbusiness \xa0– especially a newly formed one. It is very common \nfor pro fi table businesses to run short of cash. On the other \nhand, loss-making businesses can have high cash in fl ows  \nin the short term. Th e  e s s e n t i a l  d i ff erence between cash and \nprofi t can be explained with three simple examples:\nExample 1:  Jhumpa buys fresh fi sh from a market every \nday. She pays cash to the traders and gets a good deal \nbecause of this. She sells all of her stock on a high-street \nstall to shoppers who also pay her cash.\n■ In a typical week Jhumpa buys fish costing $1,000 and sells \nit for $2,000.\n■ How much profit does she make in a typical week (ignore \nother costs)? $1,000.\n■ What was the di ff erence between her cash inflows and \noutflows in a typical week? $1,000 too \xa0– as all purchases \nand sales were in cash.\n■ In this very simplified example, cash = profit at the end of \nthe week (but we have ignored other expenses too).\nCash inflows: payments in cash received by a business, \nsuch as those from customers (trade receivables) or from \nthe bank, e.g. receiving a loan.\nCash outflows: payments in cash made by a business, \nsuch as those to suppliers and workers.\nKEY TERMS\nExample 2:  Shula owns Fine Foods, a specialist \ndelicatessen. Last month she bought $500 of fresh goods \nfrom a supplier who o ff ers her one month’s credit. Th e goods sold very slowly during the month and she was \nforced to cut her prices several times. Eventually she sold \nthem all for only $300, paid in cash by her customers.\n■ What was her profit or loss (ignoring all other costs)? A loss \nof $200 \xa0– because even though she has not yet paid for the \ngoods they are still recorded as a cost.\n■ What was the di ff erence between her cash outflow and \ninflow? A positive inflow $300 \xa0– because she has not paid the \nsupplier yet. So Shula has a positive cash flow from these \ngoods this month even though she made a loss on them.\n■ Cash was not the same as profit for this business.\nExample 3:  Sanjit is concerned about competition for his \njewellery shop. He buys most of his stock over the Internet \nfor cash \xa0– but has decided to increase the credit terms \nhe gives to his customers to two months. Last month he \nbought some rings for $3,000 and paid in cash. He sold \nthem all in the same month for $7,000– yet will not receive \npayment until two months’ time.\n■ How much profit did he make on these rings? $4,000 \xa0– the \nrings have been sold and revenue recorded from the sale \neven though no cash payment has been made.\n■ What was Sanjit’s cash-flow position from these deals? \nA negative outflow of $3,000 \xa0– he may be very short of \ncash until he receives payment from his customers.\n■ Cash was not the same as profit for this business \xa0– and \nthere is a real danger that it could run out of cash to pay its \neveryday costs, such as wages and rent.\nTOP TIP\nWhen given the opportunity, emphasise the importance \nof having enough cash in the short term. Profit can wait \nto be earned in the long term \xa0– but cash payments are \nalways being made.\nHow to forecast cash flow\nForecasting cash fl ow means trying to estimate future cash \ninfl ows and cash out fl ows, usually on a month-by-month \nbasis. Let’s take the case of Mohammed, an entrepreneur \nplanning to open a car-valeting service aiming to o ff er car \ncleaning to individual customers and owners of car fl eets, \nsuch as taxi fi rms.\nForecasting cash inflows\nWhere to start? Th e business owner will probably attempt \nto forecast cash in fl ows fi rst. Some of these will be easier to \nforecast than others. Here are some example cash in fl ows \nand how they might be forecast:\n■ Owner’s own capital injection: This will be easy to forecast \nas this is under Mohammed’s direct control.46531: Forecasting and managing cash flows  ' ###""" 

prompt_template=f'''<s>[INST] {prompt} [/INST]'''
print("\n\n*** Generate:")

input_ids = tokenizer(prompt_template, return_tensors='pt').input_ids.cuda()
output = model.generate(inputs=input_ids, temperature=0.7, do_sample=True, top_p=0.95, top_k=40, max_new_tokens=400000)
print(tokenizer.decode(output[0]))
# -----------------------------------------------------------------------------

# %%
hi = pd.read_json(r'data/ready_data.json')
hi.to_csv(r'data/lemme_see.csv', escapechar='\\')
# %%
