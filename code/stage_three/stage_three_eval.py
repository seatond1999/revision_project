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
    #tokenizer.add_tokens(["<|im_start|>"])
    #tokenizer.add_special_tokens(dict(eos_token="<|im_end|>"))
    base_model.resize_token_embeddings(len(tokenizer))
    base_model.config.eos_token_id = tokenizer.eos_token_id
    base_model.config.pad_token_id = tokenizer.pad_token_id

    # Load LoRA adapter and merge
    return tokenizer, PeftModel.from_pretrained(base_model, adapter_path)


def inference(tokenizer, model, contexts, example_questions):
    outputs = []
    system = "You are an AI assisstant who will answer the question given by the user using the information provided"
    for i in example_questions:
        #prompt = (
        #    "<|im_start|>system"
        #    + f" {system}<|im_end|>"
        #    + "<|im_start|>user"
        #    + f""" question ###{i}### information ###{contexts}###<|im_end|>"""
        #    + "<|im_start|>assisstant ###"
        #)

        prompt = (
            "Context information is below.\n"
            "---------------------\n"
            f"{contexts}\n"
            "---------------------\n"
            "Given the context information and not prior knowledge, "
            "answer the query.\n"
            f"Query: {i}\n"
            f"Answer: "
        )

        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda:0")

        generation_config = GenerationConfig(
            do_sample=True,
            top_k=5,
            temperature=0.5,
            max_new_tokens=300,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
        out = model.generate(inputs=input_ids, generation_config=generation_config)
        decoded_output = tokenizer.decode(out[0], skip_special_tokens=True)
        #decoded_output = decoded_output[
        #    decoded_output.find("<|im_start|>assisstant ")
        #    + len("<|im_start|>assisstant ") :
        #]
        outputs.append((i, decoded_output))

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
    example_question1 = "Can we change the properties of protein by changing the types of bond between amino acids?"
    example_question2 = "Are there any disulfide or hydrogen bonds in my hair?"
    example_question3 = "How many types of protein structure are there?"
    example_question4 = "How many dinosaurs are still alive in America today?"
    example_questions = [
        example_question1,
        example_question2,
        example_question3,
        example_question4,
    ]
    adapter_path = "/home/seatond/revision_project/code/stage_three/prompting_answer_rank32_lr8e-06_target8_epochs1_laplha64_wuratio0.13_wdecay0.13"
    base_model_path = "TheBloke/Mistral-7B-v0.1-GPTQ"
    eval_path = r"../../evaluation_data.csv"
    context_list = pd.read_csv(eval_path)["contexts"][[10]]
    context_list = 'Hydrogen bonds break easily and reform if pH or temperature conditions change.\nDISULFIDE BONDS\nDisulfide bonds form when two cysteine molecules are close together in the structure of a polypeptide (see fig C). An oxidation reaction occurs between the two sulfur-containing groups, resulting in a strong covalent bond known as a disulfide bond. These disulfide bonds are much stronger than hydrogen bonds but they happen much less often. They are important for holding the folded polypeptide chains in place.\nhydrogen bondhydrogen bond\ndisulﬁde bonddisulﬁde bond\nα-helixα-helixα-helixα-helixβ-pleated sheetβ-pleated sheetβ-pleated sheetβ-pleated sheet\n▲ fig C Hydr ogen bonds and disulfide bonds maintain the shape of protein molecules and this determines \ntheir function.\nIONIC BONDS\nIonic bonds can form between some of the strongly positive and negative amino acid side chains \nwhich are sometimes found deep inside the protein molecules. They are strong bonds, but they are not as common as the other structural bonds.\nY our hair is made of the protein keratin.'
    example_questions = ['Under what conditions do hydrogen bonds break easily?']
    #example_questions = ['Do somatic complaints predict subsequent symptoms of depression?']
    #context_list = """Evidence suggests substantial comorbidity between symptoms of somatization and depression in clinical as well as nonclinical populations. However, as most existing research has been retrospective or cross-sectional in design, very little is known about the specific nature of this relationship. In particular, it is unclear whether somatic complaints may heighten the risk for the subsequent development of depressive symptoms.We report findings on the link between symptoms of somatization (assessed using the SCL-90-R) and depression 5 years later (assessed using the CES-D) in an initially healthy cohort of community adults, based on prospective data from the RENO Diet-Heart Study.Gender-stratified multiple regression analyses revealed that baseline CES-D scores were the best predictors of subsequent depressive symptoms for men and women. Baseline scores on the SCL-90-R somatization subscale significantly predicted subsequent self-reported symptoms of depressed mood 5 years later, but only in women. However, somatic complaints were a somewhat less powerful predictor than income and age."""
    context_list = context_list.replace("\n", " ")
    #tokenizer, model = load_model(adapter_path, base_model_path)
    outputs = inference(tokenizer, model, context_list, example_questions)
    print(outputs)

# %%
