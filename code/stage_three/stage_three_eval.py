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


def inference(tokenizer, model, contexts, example_questions):
    outputs = []
    system = "You will answer a question concisely using only the information provided by the user."
    for i in example_questions:
        prompt = (
            "<|im_start|>system"
            + f"\n{system}<|im_end|>"
            + "\n<|im_start|>user"
            + f"""Answer this question: "{i}"\nUsing only this information:"{contexts}"<|im_end|>"""
            + "\n<|im_start|>assisstant"
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
        decoded_output = decoded_output[
            decoded_output.find("<|im_start|>assisstant")
            + len("<|im_start|>assisstant") :
        ]
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
    adapter_path = "seatond/4_epochs_2"
    base_model_path = "TheBloke/Mistral-7B-v0.1-GPTQ"
    eval_path = r"../../evaluation_data.csv"
    context_list = pd.read_csv(eval_path)["contexts"][[10]]
    tokenizer, model = load_model(adapter_path, base_model_path)
    outputs = inference(tokenizer, model, context_list, example_questions)
