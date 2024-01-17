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


def prep_data(test_data_path):
    data = pd.read_json(test_data_path)
    df = pd.DataFrame(
        {
            "content": data.apply(
                lambda x: f"""<s>[INST] @@@ Instructions:
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
Assisstant: """,
                axis=1,
            ),
            'label': data['label']
        }
    )
    #return data_aq
    return df


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


def inference(tokenizer, model, data):
    generation_config = GenerationConfig(
        do_sample=True,
        top_p=0.95, top_k=40,
        temperature=0.7,
        max_new_tokens=150000,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    results = []
    for i in data['content']:
    
        out = model.generate(
            inputs=tokenizer(i, return_tensors="pt").input_ids.to("cuda:0"), 
            generation_config=generation_config
            )
        
        decoded_output = tokenizer.decode(out[0], skip_special_tokens=True)
        decoded_output = decoded_output[
                decoded_output.find("[/INST]")
                + len("[/INST]") :
            ]
        print(decoded_output)
        results.append('yes') if 'yes' in decoded_output.lower() else results.append('no')

    results_df = data.assign(prediction=results)

    return results_df

# -----------------------------------------------------------------------------


# %% --------------------------------------------------------------------------
if __name__ == "__main__":
    adapter_path = "seatond/page_identifier_rank32"
    base_model_path = "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ"
    tokenizer, model = load_model(adapter_path, base_model_path)
    test_data_path = r"firstpage_testdata.json"
    test_data = prep_data(test_data_path)
    results = inference(tokenizer,model,test_data)


