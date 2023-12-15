# %% --------------------------------------------------------------------------
import pandas as pd
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GPTQConfig,
    TrainingArguments,
    TrainerCallback,
    BitsAndBytesConfig
)
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer
import os
import tempfile
from huggingface_hub import login
from datetime import datetime as dt
import torch



# %%
def load_data():
    dataset = load_dataset(
        "pubmed_qa", "pqa_labeled"
    )  # this is the 1000 'expertly labelled' dataset (can choose from "pqa_artificial", "pqa_labeled", "pqa_unlabeled")
    data = dataset["train"].to_pandas()[["question", "context", "long_answer"]]
    data["context"] = data["context"].apply(lambda x: "".join(x["contexts"]))
    return data


data = load_data()

# %% --------------------------------------------------------------------------
def prep_data(data)    :
    data = data[0]
    system = "You will answer a question concisely using only the information provided."
    data_aq = pd.DataFrame(
            {
                "content": data.apply(
                    lambda x: "<|im_start|>system" + f'\n{system}<|im_end|>'
                    + "\n<|im_start|>user" + f"""Answer this question: "{x['question']}"\nUsing only this infomration:"{x['context']}"<|im_end|>"""
                    + "\n<|im_start|>assisstant" + f"\n{x['long_answer']}<|im_end|>",
                    axis=1,
                )
            }
        )
    data_aq = Dataset.from_pandas(data_aq)
    data_aq = data_aq.train_test_split(test_size=0.1)
    return data_aq

data = prep_data(data)
# -----------------------------------------------------------------------------
# %%
## timer
def timer(func):
    def do(x):
        before = dt.now()
        perform = func(x)
        after = dt.now()
        print(after-before)
        return perform
    return do

# %%
## timer
def timer(func):
    def do(x):
        before = dt.now()
        perform = func(x)
        after = dt.now()
        print(after-before)
        return perform
    return do

#login
token = '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'
login(token = token)

# finetune..

@timer
def finetune(data):
    token = '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'
    login(token = token)
    train_data = data["train"]
    test_data = data["test"]

    ##load model and tokenizer
    model_id = "TheBloke/Mistral-7B-v0.1-GPTQ"

    tokenizer = AutoTokenizer.from_pretrained(model_id,use_fast=False,add_eos_token = True)
    

    quantization_config_loading = GPTQConfig(
        bits=4, disable_exllama=True, tokenizer=tokenizer, 
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=quantization_config_loading, device_map="auto",torch_dtype=torch.float16
    )
    # add special tokens for chatML format ##specific to this experiment:
    tokenizer.pad_token = "</s>"
    tokenizer.add_tokens(["<|im_start|>"])
    tokenizer.add_special_tokens(dict(eos_token="<|im_end|>"))
    model.resize_token_embeddings(len(tokenizer))
    model.config.eos_token_id = tokenizer.eos_token_id

    r = 16
    lora_alpha = 16
    lr = 1.8e-4
    epochs = 3
    target_modules=["q_proj", "v_proj","k_proj"]

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
        task_type="CAUSAL_LM",
        target_modules=target_modules,
        modules_to_save = ["lm_head", "embed_tokens"] #because of special tokens #### sepcific to this experiemtn
    )
    model = get_peft_model(model, peft_config)
    print("trainable parameters:",model.print_trainable_parameters())
    ##
    name = "stage_three_TRY_3epoch"
    training_arguments = TrainingArguments(
        output_dir=name,
        per_device_train_batch_size=8, #5 works
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=1,
        optim="paged_adamw_32bit",
        learning_rate=lr,
        lr_scheduler_type="cosine",
        save_strategy="steps",  # so do we need the whole PeftSavingCallback function? maybe try withput and run the trainer(from last check=true)
        logging_steps=20,
        num_train_epochs=epochs,
        #max_steps=250,
        fp16=False,
        push_to_hub=False,
        report_to=["tensorboard"],
        group_by_length=True,
        ddp_find_unused_parameters=False,
        do_eval=True,## for eval:
        evaluation_strategy='steps',
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

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=test_data,
        peft_config=peft_config,
        dataset_text_field="content",
        args=training_arguments,
        tokenizer=tokenizer,
        callbacks=callbacks, #try if doesnt work hashing all of checkpiint stuff above and also this callback line
        packing=False,
    )

    ###################################################

    ########## set up tensorboard #####################
    tmpdir = tempfile.TemporaryDirectory()
    local_training_root = tmpdir.name

    loc_checkpoint_path = os.path.join(local_training_root, name)
    tensorboard_display_dir = f"{loc_checkpoint_path}/runs"


    #########################################################
    #trainer.train(resume_from_checkpoint='/home/seatond/revision_project/code/stage_three/stage_three_TRY') #use if want to go from checkpoint
    trainer.train()
    #trainer.state.log_history()
    #trainer.save_model()
    #trainer.push_to_hub() #un hash when want to send final model to hub
    return trainer


# resume_from_checkpoint (str or bool, optional) — If a str, local path to a saved checkpoint as saved by a previous instance of Trainer. If a bool and equals True, load the last checkpoint in args.output_dir as saved by a previous instance of Trainer. If present, training will resume from the model/optimizer/scheduler states loaded here.


# %% --------------------------------------------------------------------------
if __name__ == '__main__':
    lets_go = finetune(prep_data(load_data())) 
    lets_go.save_model()
# -----------------------------------------------------------------------------

-----------------------------------------------------------------

# %%
def load_data():
    dataset = load_dataset(
        "pubmed_qa", "pqa_labeled"
    )  # this is the 1000 'expertly labelled' dataset (can choose from "pqa_artificial", "pqa_labeled", "pqa_unlabeled")
    data = dataset["train"].to_pandas()[["question", "context", "long_answer"]]
    data["context"] = data["context"].apply(lambda x: "".join(x["contexts"]))
    return data


data = load_data()
# %%
dataset = load_dataset("pubmed_qa", "pqa_artificial")  # this is the 1000 'expertly labelled' dataset (can choose from "pqa_artificial", "pqa_labeled", "pqa_unlabeled")

# %%
dataset_keep = dataset.iloc[0:3000,:]
# %%
dataset_keep = dataset_keep[["question", "context", "long_answer"]]
# %%
dataset_keep.to_csv('synthetic_data.csv')
# %%
