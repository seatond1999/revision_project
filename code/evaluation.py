# %%
# for inference
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, GenerationConfig, GPTQConfig
import torch
from openai import OpenAI
import os
import time

# %% --------------------------------------------------------------------------

model_id_inf = "seatond/mist_question_asking_5e4LR_2epoch"

tokenizer = AutoTokenizer.from_pretrained(model_id_inf, use_fast=True)

model = AutoPeftModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=model_id_inf,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map="cuda:0",
    token="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",
)

context_list = None  # get list of test data - try with 3 points and eithe loop through each Q or ask all at once
result = []
for i in context_list:
    prompt_template = (
        f"###human: Ask me a questions about:\n{context_list}\n###Response:\n"
    )
    input_ids = tokenizer(prompt_template, return_tensors="pt").input_ids.to("cuda:0")
    quantization_config_loading = GPTQConfig(bits=4)
    generation_config = GenerationConfig(
        do_sample=True,
        top_k=1,
        temperature=0.5,
        max_new_tokens=35,
        pad_token_id=tokenizer.eos_token_id,
    )

    outputs = model.generate(inputs=input_ids, generation_config=generation_config)
    questions = (tokenizer.decode(outputs[0], skip_special_tokens=True))

    eval_template = f"""can these questions:"{questions}" be answered using only the following context: "{i}"\nIf extra information is required please start your answer with no """


# now feed question into GPT

client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": data_to_enrich.iloc[i, 0]}],
            )
            .choices[0]
            .message.content

