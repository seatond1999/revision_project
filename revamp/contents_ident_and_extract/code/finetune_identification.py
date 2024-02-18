# %% --------------------------------------------------------------------------
import pandas as pd
import numpy as np
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    GPTQConfig,
    TrainingArguments,
    TrainerCallback,
    BitsAndBytesConfig,
    Trainer,
    AutoConfig
)
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model, TaskType
from trl import SFTTrainer
import os
import tempfile
from huggingface_hub import login
from datetime import datetime as dt
import torch
from torch import nn
import tensorboard
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

# -----------------------------------------------------------------------------

# %% --------------------------------------------------------------------------
# -----------------------------------------------------------------------------


## timer
def timer(func):
    def do(*args):
        before = dt.now()
        perform = func(*args)
        after = dt.now()
        print(after - before)
        return perform

    return do


# %%
token = "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"
login(token=token)

def load():
##load model and tokenizer
    model_id = "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ"

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, use_fast=False, padding=True, truncation=True, max_length=1500
    )
    quantization_config_loading = GPTQConfig(
        bits=4,
        disable_exllama=True,
        tokenizer=tokenizer,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        quantization_config=quantization_config_loading,
        device_map="auto",
        torch_dtype=torch.float16, num_labels=2,
    )

    #model.score = nn.Identity()
    #model.config.score = nn.Identity()
    #num_classes = 2  #Adjust the number of classes according to your task
    #classification_head = nn.Linear(model.config.hidden_size, num_classes)
    #model.config.classifier = classification_head
    #model.classifier = classification_head

    tokenizer.pad_token = "<unk>"
    tokenizer.padding_side = "right"
    model.resize_token_embeddings(len(tokenizer))
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    return tokenizer,model


# Create torch dataset
class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels:
            item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])


def prep_data(tokenizer,model):
    data = pd.read_csv(r"../data/prepared_data.csv")
    df = pd.DataFrame(
        {
            "content": data.apply(
                lambda x: f"""<s>[INST] @@@ Instructions:
It is your task to classify whether a string corresponds to the contents page of a pdf book.
A contents page includes chapter titles and page numbers.
Only reply with the words "Yes" or "No"
You must reply "yes" if the string is from the contents page, and "no" if it is not the contents page.

@@@ Question:
This is the string: ### {x['contents_page']} ### [/INST]""",
                axis=1,
            ),
            'label': data.apply(lambda x: 0 if x['label']=='no' else 1,axis=1)
        }
    )

    #stratified sampling
    X = list(df['content'])
    y = list(df['label'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1,stratify=y)
    X_train_tokenized = tokenizer(X_train, padding=True, truncation=True, max_length=3000)
    X_test_tokenized = tokenizer(X_test, padding=True, truncation=True, max_length=3000)

    train_dataset = Dataset(X_train_tokenized, y_train)
    test_dataset = Dataset(X_test_tokenized, y_test)
    
    return train_dataset,test_dataset

def compute_metrics(p):
    pred, labels = p
    pred = np.argmax(pred, axis=1)

    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    recall = recall_score(y_true=labels, y_pred=pred)
    precision = precision_score(y_true=labels, y_pred=pred)
    f1 = f1_score(y_true=labels, y_pred=pred)

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


@timer
def finetune(tokenizer,model,data, r, lora_alpha, lr, epochs, target_modules,batch_s,gradacc):
    model = model
    tokenizer = tokenizer
    token = "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"
    login(token=token)
    train_data = data[0]
    test_data = data[1]

    
    r = r
    lora_alpha = lora_alpha
    lr = lr
    epochs = epochs
    target_modules = target_modules
    batch_s = batch_s
    gradacc=gradacc
    #warmup_ratio = warmup_ratio
    #wdecay=wdecay

    ##finetune:

    print(model)
    model.config.use_cache = False  # wont store intermediate states
    model.config.pretraining_tp = 1  # to replicate pre-training performance
    model.gradient_checkpointing_enable()  # compramise between forgetting activation states and remmebering them for backpropagation. Trade off computation time for GPU memory
    model = prepare_model_for_kbit_training(model)
    peft_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.SEQ_CLS,
        target_modules=target_modules,
        #modules_to_save = ['classifier']
        modules_to_save = ['classifier']
    )
    model = get_peft_model(model, peft_config)
    print("trainable parameters:", model.print_trainable_parameters())
    ##
    name = f"firstpage_c_rank{r}_lr{lr}_target{len(target_modules)}_epochs1.7_laplha{lora_alpha}_batch{batch_s}_gradacc{gradacc}" #_wuratio{warmup_ratio}_wdecay{wdecay    
    training_arguments = TrainingArguments(
        output_dir=name,
        per_device_train_batch_size=batch_s,  # 5 works
        per_device_eval_batch_size=batch_s,
        gradient_accumulation_steps=gradacc,
        optim="paged_adamw_32bit",
        learning_rate=lr,
        lr_scheduler_type="cosine",
        save_strategy="steps",  # so do we need the whole PeftSavingCallback function? maybe try withput and run the trainer(from last check=true)
        logging_steps=6,
        #save_steps=60,
        num_train_epochs=epochs,
        max_steps=85,
        fp16=False,
        push_to_hub=False,
        report_to=["tensorboard"],
        group_by_length=True,
        ddp_find_unused_parameters=False,
        do_eval=True,  ## for eval:
        evaluation_strategy="steps"
        #warmup_ratio=warmup_ratio,
        #weight_decay=wdecay,
    )

    # if checkpint doesnt work, might have to do save_steps=20 for example in the training argumnet above

    ############## checkpoint ##############################
    class PeftSavingCallback(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            checkpoint_path = os.path.join(
                args.output_dir, f"checkpoint-{state.global_step}"
            )
            kwargs["model"].save_pretrained(checkpoint_path)

            if "pytorch_model.bin" in os.listdir(checkpoint_path):
                os.remove(os.path.join(checkpoint_path, "pytorch_model.bin"))

    callbacks = [PeftSavingCallback()]

    trainer = Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=test_data,
        #peft_config=peft_config,
        #dataset_text_field="content",
        args=training_arguments,
        compute_metrics=compute_metrics,
        #packing=False,
        #max_seq_length=1500
        #max_length = 1000
        
    )

    ########## set up tensorboard #####################
    tmpdir = tempfile.TemporaryDirectory()
    local_training_root = tmpdir.name

    loc_checkpoint_path = os.path.join(local_training_root, name)
    tensorboard_display_dir = f"{loc_checkpoint_path}/runs"

    #########################################################
    # trainer.train(resume_from_checkpoint = '/home/seatond/revision_project/code/rank16_lr0.0002_target7_epochs2_laplha16/checkpoint-56')
    trainer.train()
    # trainer.state.log_history()
    # trainer.save_model()
    # trainer.push_to_hub() #un hash when want to send final model to hub
    return trainer




# %% --------------------------------------------------------------------------
# RUN!
# import os
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb=10'
if __name__ == "__main__":
    tokenizer,model = load()
    data = prep_data(tokenizer,model)
    trainer_obj = finetune(tokenizer,model,data, 16, 32, 2.2e-5, 5, ["q_proj", "v_proj","o_proj"],4,1)

#,"gate_proj"
#,"gate_proj","up_proj","down_proj"
#,"k_proj","up_proj","down_proj","gate_proj"]
# -----------------------------------------------------------------------------
# def finetune(data,r,lora_alpha,lr,epochs,target_modules,batch,,warmup_ratio,wdecay):

#,"k_proj","up_proj","down_proj","gate_proj"   
# -----------------------------------------------------------------------------

# %% --------------------------------------------------------------------------
trainer_obj.save_model() 
# -----------------------------------------------------------------------------

# %%
