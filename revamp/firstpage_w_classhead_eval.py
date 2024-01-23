# %% --------------------------------------------------------------------------
###### starting using poroper evaluation function ###################
import pandas as pd
from peft import PeftModel
from transformers import (
    AutoTokenizer,
    GenerationConfig,
    GPTQConfig,
    AutoModelForCausalLM,AutoModelForSequenceClassification
)
import torch
from openai import OpenAI
import openai
import os
import time
import tensorboard
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
#import matplotlib.pyplot as plt
 


def prep_data(test_data_path):
    data = pd.read_json(test_data_path)
    df = pd.DataFrame(
        {
            "content": data.apply(
                lambda x: f"""<s>[INST] This is a string from a page of a pdf book: ### {x['page']} ###
                Is it true or false that this page belongs to a chapter called: ### {x['chapter_title']} ###? [/INST]""",
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

    tokenizer = AutoTokenizer.from_pretrained(
        base_path, use_fast=False, padding=True, truncation=True, max_length=1500
    )
    quantization_config_loading = GPTQConfig(
        bits=4,
        disable_exllama=True,
        tokenizer=tokenizer,
    )

    base_model = AutoModelForSequenceClassification.from_pretrained(
        base_path,
        quantization_config=quantization_config_loading,
        device_map="auto",
        torch_dtype=torch.float16, num_labels=2
    )


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
        if 'yes' in decoded_output.lower():
            results.append('yes')
        elif 'no' in decoded_output.lower():
            results.append('no')
        else:
            results.append('error')

    results_df = data.assign(prediction=results)

    return results_df

def metrics(df):
    conf_matrix = confusion_matrix(df['label'], df['prediction'])
    conf_matrix_df = pd.DataFrame(conf_matrix, index=['Actual No', 'Actual Yes'], columns=['Predicted No', 'Predicted Yes'])

    # Plot the confusion matrix using seaborn heatmap
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()

    classes = ['No', 'Yes']
    tick_marks = range(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    for i in range(len(classes)):
        for j in range(len(classes)):
            plt.text(j, i, str(conf_matrix[i, j]), ha='center', va='center', color='black', fontsize=14)
    plt.show()

    recall = conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[1, 0]) #check these calcs are right
    specificity = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1])
    return recall,specificity
# -----------------------------------------------------------------------------


# %% --------------------------------------------------------------------------
if __name__ == "__main__":
    adapter_path = "/home/seatond/revision_project/revamp/firstpage_rank32_lr2.2e-05_target7_epochs1_laplha64_batch2_gradacc2"
    base_model_path = "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ"
    tokenizer, model = load_model(adapter_path, base_model_path)
    test_data_path = r"firstpage_testdata.json"
    test_data = prep_data(test_data_path)
    results = inference(tokenizer,model,test_data)
    recall, specificity = metrics(results)



# %% --------------------------------------------------------------------------
#confusion matrix and recall and specificity

df = pd.DataFrame({'label': ['yes', 'no', 'yes', 'no', 'yes'],'prediction': ['yes', 'no', 'yes', 'yes', 'yes']})
conf_matrix = confusion_matrix(df['label'], df['prediction'])
conf_matrix_df = pd.DataFrame(conf_matrix, index=['Actual No', 'Actual Yes'], columns=['Predicted No', 'Predicted Yes'])

# Plot the confusion matrix using seaborn heatmap
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()

classes = ['No', 'Yes']
tick_marks = range(len(classes))
plt.xticks(tick_marks, classes)
plt.yticks(tick_marks, classes)

plt.xlabel('Predicted Label')
plt.ylabel('True Label')
for i in range(len(classes)):
    for j in range(len(classes)):
        plt.text(j, i, str(conf_matrix[i, j]), ha='center', va='center', color='black', fontsize=14)
plt.show()

recall = conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[1, 0]) #check these calcs are right
specificity = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1])
# -----------------------------------------------------------------------------

# %%
adapter_path = "/home/seatond/revision_project/revamp/firstpage_rank32_lr2.2e-05_target7_epochs1_laplha64_batch2_gradacc2"
base_model_path = "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ"
tokenizer, model = load_model(adapter_path, base_model_path)
test_data_path = r"firstpage_testdata.json"
test_data = prep_data(test_data_path)
# %%
ground = []
pred = []
for i in range(0,38):
    print(i)
    X_test_tokenized = tokenizer(test_data['content'][i], return_tensors="pt").input_ids.to("cuda:0")
    with torch.no_grad():
        logits = model(X_test_tokenized).logits

    # Convert logits to probabilities using softmax
    probs = torch.nn.functional.softmax(logits, dim=-1)

    # Get the predicted class index
    predicted_class = torch.argmax(probs).item()
    predicted_class = 'yes' if predicted_class == 1 else 'No'

    pred.append(predicted_class)
    ground.append(test_data['label'][i])
    print(i)

# %%
results = pd.DataFrame({'ground':ground,'pred':pred})
results

# %%
