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
    prompt = f"""<s>[INST] @@@ Instructions:
You are an assisstant who must classify whether a string from a page of a pdf book corresponds to the first page of a given chapter in that book.
You will be given the string and also given the chapter title.
The first page of a chapter usually contains the name of the chapter towards the start of the string.
The first word of you answer must be "Yes" or "No"
You must reply "yes" if the string is the first page of the given chapter, and "no" if it is not the first page of the given chapter.

@@@ Example:
User: The given chapter title is ### The Chemistry Of Life ### and the string is: ### 'CHEMISTRY FOR BIOLOGISTS 61A.1 THE CHEMISTRY OF LIFE\nTHE CHEMISTRY OF WATER\nAll reactions in living cells take place in water. Without water, \nsubstances could not move around the body. Water is one of  the reactants in the process of  photosynthesis, on which almost all life depends (see fig E). Understanding the properties of  water will help you understand many key systems in living organisms. \nWater is also a major habitat – it supports more life than any other \npart of  the planet.\n▲ fig E  W ater is vital for life on Earth in many different ways – in a desert, \nthe smallest amount of water allows plants to grow.\nThe simple chemical formula of  water is H2O. This tells us that \ntwo atoms of  hydrogen are joined to one atom of  oxygen to make \nup each water molecule. However, because the electrons are held closer to the oxygen atom than to the hydrogen atoms, water is a polar molecule (see fig F).\n104.5°Oδ2\nHδ1Hδ1\n▲ fig F  A model of a w ater molecule showing dipoles.\nOne major effect of  this polarity is that water molecules form hydrogen  bonds. The slightly negative oxygen atom of  one water \nmolecule will attract the slightly positive hydrogen atoms of  other water molecules in a weak electrostatic attraction called a hydrogen \nbond. Each individual hydrogen bond is weak but there are many of  them so the molecules of  water ‘stick together’ more than you might expect (see fig G). Water has relatively high melting and boiling points compared with other substances that have molecules of  a similar size because it takes a lot of  energy to break all the hydrogen bonds that hold the molecules together. Hydrogen bonds are important in protein structure (see Sections 1A.5 and 2B.1) \nand in the structure and functioning of  DNA (see Section 2B.3).Oδ2\nOδ2\nHδ1Hδ1Hδ1Hδ1\nHδ1Oδ2Oδ2\nOδ2Hδ1\nHδ1\nHδ1Hδ1Hδ1\n▲ fig G  Hydr ogen bonding in water molecules, based on attraction \nbetween positive and negative dipoles.\nTHE IMPORTANCE OF WATER\nThe properties of  water make it very important in biological \nsystems for many reasons.\n •W ater is a polar solvent. Because it is a polar molecule, many ionic \nsubstances like sodium chloride will dissolve in it (see fig H).  \nMany covalently bonded substances are also polar and will dissolve in water, but often do not dissolve in other covalently bonded solvents such as ethanol. Water also carries other substances, such as starch. As a result, most of  the chemical reactions within cells occur in water (in aqueous solution).\nsodium and chloride ionsin solution in water\nsalt and water mixed\nsodium chloride\nNaClionic bond sodium ionchlorideionδ1 chargeson hydrogenin water areattracted tonegativechloride ion\nδ2\n charges\non oxygenin water ar\ne\nattracted tothe positivesodium ionH\nH\nHHH\nH HO\nOO\nCl2\nCl2\nCl2Cl2\nCl2\nNa1OH\nCl2\nNa1Na1\nCl2Cl2\nCl2\nCl2Cl2\nNa1Na1Na1 Na1\nNa1Na1Na1\n▲ fig H  A model of sodium chloride dissolving in water as a result of the \ninteractions between the charges on sodium and chloride ions and the dipoles of the water molecules.\nUncorrected proof, all content subject to change at publisher discretion. Not for resale, circulation or distribution in whole or in part. ©Pearson 2018' ###
Assisstant: "Yes"

@@@ Example:
User: The given chapter title is ### Preparing For Your Exams ### and the string is: ### 'xASSESSMENT OVERVIEW\nPAPER / UNIT 1PERCENTAGE \nOF IASPERCENTAGE OF IALMARK TIME AVAILABILITY\nMOLECULES, DIET, TRANSPORT AND \nHEALTH \nWritten examination\nPaper code \nWBI11/01\nExternally set and marked by \nPearson Edexcel\nSingle tier of entry40% 20% 80 1 hour  \n30 minutesJanuary, June and October\nFirst assessment : January 2019\nPAPER / UNIT 2PERCENTAGE \nOF IASPERCENTAGE OF IALMARK TIME AVAILABILITY\nCELLS, DEVELOPMENT, BIODIVERSITY \nAND CONSERVATION\nWritten examination\nPaper code \nWBI12/01\nExternally set and marked by \nPearson Edexcel\nSingle tier of entry40% 20% 80 1 hour  \n30 minutesJanuary, June and October\nFirst assessment : June 2019\nPAPER / UNIT 3PERCENTAGE \nOF IASPERCENTAGE OF IALMARK TIME AVAILABILITY\nPRACTICAL SKILLS IN BIOLOGY 1   \nWritten examination\nPaper code \nWBI13/01\nExternally set and marked by \nPearson Edexcel\nSingle tier of entry20% 10% 50 1 hour  \n20 minutesJanuary, June and October\nFirst assessment : June 2019ASSESSMENT OVERVIEW\nThe following tables give an overview of the assessment for Pearson Edexcel International Advanced Subsidiary course \nin Biology. You should study this information closely to help ensure that you are fully prepared for this course and know exactly what to expect in each part of the examination. More information about this qualification, and about the question types in the different papers, can be found on page 302 of this book.\nUncorrected proof, all content subject to change at publisher discretion. Not for resale, circulation or distribution in whole or in part. ©Pearson 2018' ###
Assisstant: "No"

@@@ Question:
User: The given chapter title: ### {x['chapter_title']} ### This is the string: ### {x['page']} ### [/INST]
Assisstant: """

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
    base_model_path = "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ"
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
# 
# -----------------------------------------------------------------------------
