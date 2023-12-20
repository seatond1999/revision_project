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


def load_model(lora_adapters, base_model):
    base_path = base_model  # input: base model
    adapter_path = lora_adapters  # input: adapters

    base_model = AutoModelForCausalLM.from_pretrained(
        base_path,
        return_dict=True,
        device_map="auto",
    )

    tokenizer = AutoTokenizer.from_pretrained(base_path)

    # Add/set tokens (same 5 lines of code we used before training)
    tokenizer.pad_token = "</s>"
    tokenizer.add_tokens(["<|im_start|>"])
    tokenizer.add_special_tokens(dict(eos_token="<|im_end|>"))
    base_model.resize_token_embeddings(len(tokenizer))
    base_model.config.eos_token_id = tokenizer.eos_token_id
    base_model.config.pad_token_id = tokenizer.pad_token_id

    # Load LoRA adapter and merge
    return tokenizer, PeftModel.from_pretrained(base_model, adapter_path)


def inference(tokenizer, model, contexts):
    outputs = {}
    sys = "You are a question-asking AI assisstant who will ask concise questions which can be answered using only the information provided by the user."
    for i, j in enumerate(contexts):
        print(i)
        j = j.replace('\n',' ')
        j = j.replace('\xa0–',' ')
        j = j.replace('\xa0',' ')
        prompt = (
            "<|im_start|>system "
            + f"{sys}<|im_end|>"
            + "<|im_start|>user"
            + f" Information ###{j}###<|im_end|>"
            + "<|im_start|>assisstant ###"
        )

        
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda:0")

        generation_config = GenerationConfig(
            do_sample=True,
            top_k=5,
            temperature=0.2,
            max_new_tokens=150,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
        out = model.generate(inputs=input_ids, generation_config=generation_config)
        decoded_output = tokenizer.decode(out[0], skip_special_tokens=True)
        decoded_output = decoded_output[
            decoded_output.find("<|im_start|>assisstant")
            + len("<|im_start|>assisstant") :
        ]
        outputs[i] = (j, decoded_output)

    return outputs


def score_with_gpt(outputs):
    os.environ["OPENAI_API_KEY"] = "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"
    key = "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"
    openai.api_key = key
    scores = []
    counter = 0
    for i in outputs:
        time.sleep(60) if (counter) % 3 == 0 and counter != 0 else None
        counter += 1
        questions = list(map(lambda x: x + "?", outputs[i][1].split("?")))
        questions = questions[: len(questions) - 1]
        for j in questions:
            print(j, outputs[i][0][:50])
            client = OpenAI()
            scores.append(
                client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        # {'role':'system','content':'you will respond with the word "yes" or "no" only. If the answer to a question is not available in the information provided, you will respond with "no", otherwise you iwll respond with "no".'},
                        {
                            "role": "user",
                            "content": f'''can the following question be answered using only the provided context? Question: ### {j} ### \n Context: ### {outputs[i][0]} ### \n if any extra information is needed to answer the question please respond with the word "no". If it can be answered, please respond with the word "yes."''',
                        }
                    ],
                )
                .choices[0]
                .message.content
            )

    return scores


# -----------------------------------------------------------------------------


# %% --------------------------------------------------------------------------
if __name__ == "__main__":
    adapter_path = "/home/seatond/revision_project/code/CHOSEN_rank32_lr1.5e-05_target8_epochs1_laplha64_wuratio0.14_wdecay0.14"
    base_model_path = "TheBloke/Mistral-7B-v0.1-GPTQ"
    eval_path = r"../evaluation_data.csv"
    context_list = list(pd.read_csv(eval_path)["contexts"])
    tokenizer, model = load_model(adapter_path, base_model_path)
    outputs = inference(tokenizer, model, context_list)
    #scores = score_with_gpt(outputs)
    #score = (len(list(filter(lambda x: x.lower() == "yes", scores)))) / len(scores)
    #print(score)
    #print(
    #single = inference(tokenizer, model, list(pd.read_csv(eval_path)["contexts"][[10]]))[0]
    #)  # this is just for 1 row of eval table which is the usual context for comparison

# %%specific stuff:
def specific_inference(tokenizer, model, context):
    sys = "You are a question-asking AI assisstant who will ask concise questions which can be answered using only the information provided by the user."
    prompt = (
            "<|im_start|>system "
            + f"{sys}<|im_end|>"
            + "<|im_start|>user"
            + f" Information ###{context}###<|im_end|>"
            + "<|im_start|>assisstant ###"
        )
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda:0")

    generation_config = GenerationConfig(
        do_sample=True,
        top_k=5,
        temperature=0.2,
        max_new_tokens=200,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    out = model.generate(inputs=input_ids, generation_config=generation_config)
    decoded_output = tokenizer.decode(out[0], skip_special_tokens=True)
    decoded_output = decoded_output[
        decoded_output.find("<|im_start|>assisstant")
        + len("<|im_start|>assisstant") :
    ]
    return decoded_output


# %% --------------------------------------------------------------------------
if __name__ == "__main__":
    
    adapter_path = "/home/seatond/revision_project/code/NEW_REMn_rank16_lr2e-05_target8_epochs2_laplha32_wuratio0.125_wdecay0.25/checkpoint-120"
    base_model_path = "TheBloke/Mistral-7B-v0.1-GPTQ"
    tokenizer, model = load_model(adapter_path, base_model_path)

# -----------------------------------------------------------------------------


# %%
context_apm = 'Phosphatidylethanolamine N-methyltransferase (PEMT), a liver enriched enzyme, is responsible for approximately one third of hepatic phosphatidylcholine biosynthesis. When fed a high-fat diet (HFD), Pemt(-/-) mice are protected from HF-induced obesity; however, they develop steatohepatitis. The vagus nerve relays signals between liver and brain that regulate peripheral adiposity and pancreas function. Here we explore a possible role of the hepatic branch of the vagus nerve in the development of diet induced obesity and steatohepatitis in Pemt(-/-) mice.", "8-week old Pemt(-/-) and Pemt(+/+) mice were subjected to hepatic vagotomy (HV) or capsaicin treatment, which selectively disrupts afferent nerves, and were compared to sham-operated or vehicle-treatment, respectively. After surgery, mice were fed a HFD for 10 weeks.", "HV abolished the protection against the HFD-induced obesity and glucose intolerance in Pemt(-/-) mice. HV normalized phospholipid content and prevented steatohepatitis in Pemt(-/-) mice. Moreover, HV increased the hepatic anti-inflammatory cytokine interleukin-10, reduced chemokine monocyte chemotactic protein-1 and the ER stress marker C/EBP homologous protein. Furthermore, HV normalized the expression of mitochondrial electron transport chain proteins and of proteins involved in fatty acid synthesis, acetyl-CoA carboxylase and fatty acid synthase in Pemt(-/-) mice. However, disruption of the hepatic afferent vagus nerve by capsaicin failed to reverse either the protection against the HFD-induced obesity or the development of HF-induced steatohepatitis in Pemt(-/-) mice.'
context_apm = 'To date, no data is available about procalcitonin (PCT) levels and its relevance to morbidity and graft function in the early phase after pediatric liver transplantation (pLTx). The aim of this study was to analyse the prognostic relevance of early postoperative PCT elevations in pediatric liver recipients.", "Thirty pediatric patients who underwent 32 liver transplantations were included into this observational single-center study.", "Patients with high PCT levels on postoperative day (POD) 2 had higher International Normalized Ratio values on POD 5 (p<0.05) and suffered more often from primary graft non-function (p<0.05). They also had a longer stay in the pediatric intensive care unit (p<0.01) and on mechanical ventilation (p=0.001). There was no correlation between PCT elevation and systemic infection. However, PCT levels were correlated with peak serum lactate levels immediately after graft reperfusion and elevation of serum aminotransferases on POD 1 (r2=0.61, p<0.001).'
context_bio = pd.read_csv(r'../evaluation_data.csv')['contexts'][6]
test = '17 1A.5 PROTE INS CHEMIS TRY FOR BIOLOGIS TS\nHYDROGEN BONDS\nY ou were introduced to hydrogen bonds in Section 1A.1. These same bonds are essential in prot ein \nstructures. In ami no acids, tiny nega tive charges are present on the oxygen of  the carboxyl groups and tiny positive charges are present on the hydrogen atoms of  the amino groups. When these charged groups are close to each other, the opposite charges attract, forming a hydrogen bond. Hydrogen bonds are weak but, potentially, they can be made between any two amino acids in the co rrect position, so there are ma ny of  them holding the protein together very firmly. They are very important in the folding and coiling of  polypeptide cha ins (see fig C). Hydr ogen bo nds break eas ily and reform if  pH or temperature conditions change.\nDISULFIDE BONDS\nDisulfide bonds form when two cysteine molecules are close together in the structure of  a polypeptide (see fig C). An oxidation reaction occurs between the two sulfur-containing groups, resulting in a strong covalent bond known as a disulfide bond. These disulfide bonds are much stronger than hydrogen bonds but they happen much less often. They are important for holding the folded polypeptide chains in place.\nhydrogen bondhydrogen bond\ndisulﬁde bonddisulﬁde bond\nα-helixα-helixα-helixα-helixβ-pleated sheetβ-pleated sheetβ-pleated sheetβ-pleated sheet\n▲ fig C  Hydr ogen bonds and disul fide bonds maint ain the shape of prot ein molecules and this determines  \ntheir function.\nIONIC BONDS\nIonic bonds can form between some of  the strongly positive and negative amino acid side chains \nwhich are sometimes found deep inside the protein molecules. They are strong bonds, but they are not as common as the other structural bonds.\nY our hair is made of  the protein keratin. Some methods of  styling hair change the bonds within the protein \nmolecules. Blow drying or straightening hair breaks the hydrogen bonds and temporarily reforms them with the hair curl ing in a differ ent way until the hydr ogen bonds reform in their original places.\nPerming is a chemical treatment which is used in some hair salons to completely change the way hair \nlooks for weeks or months. The chemicals break the disulfide bonds between the polypeptide chains and reform them in a different place. This effect is permanent – hair will stay styled in that particular way until it is cut off.\nPROTEIN STRUCTURE\nProteins can be described by their primary, secondary, tertiary and quaternary structure (see fig\xa0D). \n •T\nhe primary structure of  a protein is the sequence of  amino acids that make up the polypeptide \nchain, held together by peptide bonds.\n •T\nhe secondary structure of  a protein is the arrangement of  the polypeptide chain into a regular, \nrepeating three-dimen sional (3D) structure, held tog ether by hydrogen bonds. One example is the right-ha nded helix (α-helix), a spi ral coil with the peptide bonds forming the backbone and the R\xa0groups protruding in all directions. An other is the ß-pleated sheet, in wh ich the pol ypeptide chain folds into reg ular pleats held together by hydrogen bonds between the amino and carbo xyl ends of  the amino acids. Most fibrous prot eins have this ty pe of  structure. Some times there is no regular secondary structure and the polypep tide forms a random coil.LEARNING TIP\nRemember that fibrous proteins \nhave a simpler structure and so tend to be more stable to changes in temperature and pH.\nUncorrected proof, all content subject to change at publisher discr etion. Not for resale,  circulation or distribution in whole or in part. ©Pearson 2018'
test2 = test.replace('\xa0–', '')
test2 = test.replace('\xa0', '')
test2 = test2.replace('\n',' ')
outputs = specific_inference(tokenizer, model, test2)
print(outputs)
# %%
pd.read_csv(r'../evaluation_data.csv')['contexts'][10]
# %%
hi = "<|im_start|>system " + f"sys_message<|im_end|>" + "<|im_start|>user"+f" Information ###INFO###<|im_end|>"+ "<|im_start|>assisstant "+ f"###QUESTIONS###<|im_end|>"
# %%
import json

# Specify the file path
file_path = 'output_fixed_prompt.txt'

# Save the dictionary to a text file
with open(file_path, 'w') as file:
    json.dump(outputs, file)
# %%
eval_path = r"../evaluation_data.csv"
pd.read_csv(eval_path)["contexts"][9]
#print(pd.read_csv(eval_path)["contexts"][3])
# %%
