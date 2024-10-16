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


# %% --------------------------------------------------------------------------
import fitz 

def read_pdf(file_path,page_number):
    with fitz.open(file_path) as pdf_document:
        page = pdf_document[page_number]
        text = page.get_text()

    return text
# -----------------------------------------------------------------------------


def load_model(lora_adapters, base_model):
    base_path = base_model  # input: base model
    adapter_path = lora_adapters  # input: adapters
    tokenizer = AutoTokenizer.from_pretrained(base_path)

    tokenizer.pad_token = "<unk>"
    tokenizer.padding_side = "right"
    quantization_config_loading = GPTQConfig(
            bits=4,
            #disable_exllama=True, #trying hashing out in finetune to match inference as reduces latency.
            tokenizer=tokenizer,
        )
    base_model = AutoModelForCausalLM.from_pretrained(
        base_path,
        return_dict=True,
        device_map="auto",
        quantization_config=quantization_config_loading,
        torch_dtype=torch.float16
    )
    
    
    base_model.resize_token_embeddings(len(tokenizer))
    base_model.config.eos_token_id = tokenizer.eos_token_id
    base_model.config.pad_token_id = tokenizer.pad_token_id

    # Load LoRA adapter and merge
    #return tokenizer,base_model
    return tokenizer, PeftModel.from_pretrained(base_model, adapter_path)


def inference(tokenizer, model, contents_page):
    #context = context.replace('\n',' ')
    #context = context.replace('\xa0–',' ')
    #context = context.replace('\xa0',' ')
    prompt = f"""<s>[INST] @@@ Instructions:
It is your task to extract the chapters and corresponding page numbers from a string which was created from the contents page of a pdf book.
You must return a list of the chapters and page numbers.
Put each chapter and its page number on its own line, and separate chapters titles from page numbers with a "---".
You will be penalised for separating chapters with anything that is not "---"
For example the first 2 chapters of a contents page should be in the following format: "chapter 1 title --- chapter 1 page number \n chapter 2 title --- chapter 2 page number"

@@@ Question:
string which was created from the contents page of a pdf book: ### {contents_page} ### [/INST]

Output:  """

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
    adapter_path = "seatond/EXTRACTION_rank32_lr2.2e-05_target7_epochs1.7_laplha64_batch1_gradacc4"
    base_model_path = "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ"
    eval_path = r'../data/test_data.csv'
    context_list = list(pd.read_csv(eval_path).iloc[:]['test_set'])
    tokenizer, model = load_model(adapter_path, base_model_path)
    #test_data = pd.read_csv(r'../data/evaluation_data.csv').iloc[1:2]
    outputs = []
    counter = 1
    for i in context_list:
    #for i in ['This chapter covers syllabus \nsection AS Level 1.1.\nDULIP STARTS HIS BUSINESS\nDulip lives in a large country with many natural \nresources, such as coal and timber. He plans to start \na business growing and cutting trees to sell as timber. \nHe wants to buy a forest from a farmer and cut down a \nfixed number of trees each year. As Dulip is concerned \nabout his environment country’s, he will plant two new \ntrees for each one he cuts down. He has been planning \nthis business venture for some time. He has visited a \nbank to arrange a loan. He has contacted suppliers of \nsaws and other equipment to check on prices. Dulip \nhas also visited several furniture companies to see \nif they would be interested in buying wood from the \nforest. In fact, he did a great deal of planning before he \nwas able to start his business.\nDulip is prepared to take risks and will invest his \nown savings, as well as using the bank loan, to set \nup the business. He plans to employ three workers \nto help him to start with. If the business is a success, \nthen he will also try to sell some of the timber abroad. \nHe knows that timber prices are high in some foreign \nmarkets. A ft er several months of planning, he was able \nto purchase the forest.\nPoints to think about:\n■ Why do you think Dulip decided to own and run his own \nbusiness rather than work for another firm?\n■ Why was it important to Dulip that he should do so much \nplanning before starting his business?\n■ D o  y o u  t h i n k  D u l i p  w i l l  m a k e  a  s u c c e s s f u l  e n t r e p r e n e u r ?\n■ Are new businesses such as Dulip’s good for a country’s \neconomy?1 Enterprise\nIntroducing the topic\nIntroduction\nMany business managers are paid high salaries to take risks \nand make decisions that will in fl uence the future success \nof their business. Much of this book is concerned with \nhow these decisions are made, the information needed to \nmake them and the techniques that can assist managers in \nthis important task. However, no student of Business can \nhope to make much progress in the study of this subject \nunless they have a good understanding of the economic \nenvironment in which a business operates. Business activity \ndoes not take place in isolation from what is going on \naround it. Th e very structure and health of the economy \nwill have a great impact on how successful business activity \nis. Th e central purpose of this whole unit, ‘Business and \nits environment’, is to introduce the inter-relationships \nbetween businesses, the world in which they operate and the \nlimits that governments impose on business activity. Th i s  \nfi rst chapter explains the nature of business activity and the \nrole of enterprise in setting up and developing businesses.\nThe purpose of business activity\nA business is any organisation that uses resources to \nmeet the needs of customers by providing a product or \nservice that they demand. Th ere are several stages in the \nproduction of fi nished goods. Business activity at all stages \ninvolves creating and adding value to resources, such \nas raw materials and semi- fi nished goods, and making \nthem more desirable to \xa0– and thus valued by \xa0– the fi nal \npurchaser. Without business activity, we would all still be \nentirely dependent on the goods that we could make or \ngrow ourselves \xa0– as some people in virtually undiscovered \nnative communities still are. Business activity uses the \nscarce resources of our planet to produce goods and \nservices that allow us all to enjoy a very much higher \nstandard of living than would be possible if we remained \nentirely self-su ffi  cient.On completing this chapter, you will be able to:\n■ understand what business activity involves\n■ recognise that making choices as a result of the \n‘economic problem’ always results in opportunity cost\n■ analyse the meaning and importance of creating value\n■ recognise the key characteristics of successful \nentrepreneurs■ assess the importance of enterprise and entrepreneurs \nto a country’s economy\n■ understand the meaning of social enterprise and the \ndiff erence between this and other businesses.\n3']:
        print(counter)
        outputs.append(inference(tokenizer, model, i))
        print(counter)
        counter +=1
    #outputs = inf_on_all(test_data)


print(outputs)


# %% --------------------------------------------------------------------------

pdf_path = r'../../../../business book for testing.pdf'
page = 3
conents_page = read_pdf(pdf_path,page) 
adapter_path = "seatond/EXTRACTION_rank32_lr2.2e-05_target7_epochs1.7_laplha64_batch1_gradacc4"
base_model_path = "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ"
tokenizer, model = load_model(adapter_path, base_model_path)
inference(tokenizer, model, conents_page)
# -----------------------------------------------------------------------------
