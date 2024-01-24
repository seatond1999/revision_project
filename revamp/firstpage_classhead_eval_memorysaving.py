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
from torch import nn
from openai import OpenAI
import openai
import os
import time
import tensorboard
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
#import matplotlib.pyplot as plt
 

# %% --------------------------------------------------------------------------
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


def load_model(lora_adapters, base_model,cond):
    base_path = base_model  # input: base model
    adapter_path = lora_adapters  # input: adapters

    tokenizer = AutoTokenizer.from_pretrained(base_path)
    tokenizer.pad_token = "<unk>"
    tokenizer.padding_side = "right"
    

    quantization_config_loading = GPTQConfig(
        bits=4,
        disable_exllama=True,
        tokenizer=tokenizer,
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_path,
        return_dict=True,
        quantization_config=quantization_config_loading,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    ###### adjusting model and model config of causal to match the automdelforseqclass which finetuned classfiication on ###########

    model.num_labels = 2
    model.config.num_labels = 2

    model.score = nn.Linear(model.config.hidden_size, model.config.num_labels, bias=True,dtype=torch.float16)

    model.lm_head = None
    model.config.lm_head = None  #remove mpapping to vocab size

    #remove mpapping to vocab size
    #removing above heads nut in actual thing would save them and then put them back for causal bits and remove score head

    #num_classes = 2  #yes or no as labels
    #classification_head = nn.Linear(model.config.hidden_size, num_classes,bias=False) #create linear layer which will add on

    #model.config.score = classification_head.state_dict() #stops the error Object of type Linear is not JSON serializable
    #model.config.score["weight"] = model.config.score["weight"].tolist() # solving the issue of not being able to JSON serialize due to adding own linear layer.
    #model.score = classification_head #add linear layer mapping to my 2 classes instead of 32k vocab 
    model.conditioner = cond
    model.resize_token_embeddings(len(tokenizer))
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    #return tokenizer,model
    return tokenizer, PeftModel.from_pretrained(model, adapter_path) #combine with adapters

def inference(tokenizer,model,test_data):
    ground = []
    pred = []
    model = model.to("cuda:0")
    for i in range(0,len(test_data)):
        print(i)
        X_test_tokenized = tokenizer(test_data['content'][i], return_tensors="pt").input_ids.to("cuda:0")

        X_test_tokenized = X_test_tokenized.to("cuda:0")
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
    return pd.DataFrame({'ground':ground,'pred':pred})

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

# -----------------------------------------------------------------------------


# %% --------------------------------------------------------------------------
if __name__ == "__main__":
    adapter_path = "/home/seatond/revision_project/revamp/firstpage_c_rank32_lr2.3e-05_target7_epochs1.7_laplha64_batch4_gradacc1"
    base_model_path = "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ"
    tokenizer, model = load_model(adapter_path, base_model_path,'classification')
    test_data_path = r"firstpage_testdata.json"
    test_data = prep_data(test_data_path)
    results = inference(tokenizer,model,test_data)
    #recall, specificity = metrics(results)




# %% --------------------------------------------------------------------------
#input_ids = tokenizer('hello there', return_tensors="pt").input_ids.to("cuda:0")

#model = model.to("cuda:0")

# Move input tensors to GPU
#input_ids = input_ids.to("cuda:0")

#with torch.no_grad():
#    logits = model(input_ids).logits

#probs = torch.nn.functional.softmax(logits, dim=-1) # hashing out as only 2 classes dont need to softmax

# Get the predicted class index
#predicted_class = torch.argmax(logits).item()
#predicted_class = 'yes' if predicted_class == 1 else 'No'
#predicted_class
# -----------------------------------------------------------------------------

#results.to_csv(r'merge_class_to_causal.csv')
# %%
