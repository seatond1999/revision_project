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
import tensorboard


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
def prep_data(data):
    system = "You are an AI assisstant who will answer a question using the context information provided and not prior knowledge"
    #data_aq = pd.DataFrame(
    #        {
    #            "content": data.apply(
    #                lambda x: "<|im_start|>system" + f' {system}<|im_end|>'
    #               + "<|im_start|>user" + f""" this is the context information ###{x['context']}### Given the context information and not prior knowledge, answer the  ###{x['question']}###<|im_end|>"""
    #                + "<|im_start|>assisstant " + f"answer ###{x['long_answer']}###<|im_end|>",
    #                axis=1,
    #            )
    #        }
    #    )
    data_aq = pd.DataFrame(
            {
                "content": data.apply(
                    lambda x: "Context information is below.\n"
                    "---------------------\n"
                    f"{x['context']}\n"
                    "---------------------\n"
                    "Given the context information and not prior knowledge, "
                    "answer the query.\n"
                    f"Query: {x['question']}\n"
                    f"Answer: {x['long_answer']}",
                    axis=1,
                )
            }
        )
    

    
    data_aq = Dataset.from_pandas(data_aq)
    data_aq = data_aq.train_test_split(test_size=0.05)
    return data_aq

data = prep_data(data)

def finetune(data,name, r, lora_alpha, lr, epochs, target_modules,warmup_ratio,wdecay):
    token = "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"
    login(token=token)
    train_data = data["train"]
    test_data = data["test"]

    ##load model and tokenizer
    model_id = "TheBloke/Mistral-7B-v0.1-GPTQ"

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, use_fast=False
    )

    quantization_config_loading = GPTQConfig(
        bits=4,
        disable_exllama=True,
        tokenizer=tokenizer,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quantization_config_loading,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    # add special tokens for chatML format ##specific to this experiment:
    tokenizer.pad_token = "</s>"
    #tokenizer.add_tokens(["<|im_start|>"])
    #tokenizer.add_special_tokens(dict(eos_token="<|im_end|>"))
    tokenizer.padding_side = "right"
    model.resize_token_embeddings(len(tokenizer))
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    r = r
    lora_alpha = lora_alpha
    lr = lr
    epochs = epochs
    target_modules = target_modules
    warmup_ratio = warmup_ratio
    wdecay=wdecay

    ##finetune:

    print(model)
    model.config.use_cache = False  # wont store intermediate states
    model.config.pretraining_tp = 1  # to replicate pre-training performance
    model.gradient_checkpointing_enable()  # compramise between forgetting activation states and remmebering them for backpropagation. Trade off computation time for GPU memory
    model = prepare_model_for_kbit_training(model)
    peft_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=0.15,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
        modules_to_save=[
            "lm_head",
            "embed_tokens",
        ],  # because of special tokens #### sepcific to this experiemtn
    )
    model = get_peft_model(model, peft_config)
    print("trainable parameters:", model.print_trainable_parameters())
    ##
    
    #name = "5_epochs"
    training_arguments = TrainingArguments(
        output_dir=name,
        per_device_train_batch_size=9,  # 5 works
        per_device_eval_batch_size=9,
        gradient_accumulation_steps=1,
        optim="paged_adamw_32bit",
        learning_rate=lr,
        lr_scheduler_type="cosine",
        save_strategy="steps",
        save_steps=1180, ##change if want a checkpoint 
        logging_steps=15,
        num_train_epochs=epochs,
        # max_steps=250,
        fp16=False,
        push_to_hub=False,
        report_to=["tensorboard"],
        group_by_length=True,
        ddp_find_unused_parameters=False,
        do_eval=True,  ## for eval:
        evaluation_strategy="steps",
        warmup_ratio=warmup_ratio,
        weight_decay=wdecay,
        #load_best_model_at_end = True,
        #metric_for_best_model="eval_loss" #adding early stopping when overfits
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

    #callbacks = [PeftSavingCallback(),EarlyStoppingCallback(early_stopping_patience=1, early_stopping_threshold=0.0000000001)]
    callbacks = [PeftSavingCallback()]

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=test_data,
        peft_config=peft_config,
        dataset_text_field="content",
        args=training_arguments,
        tokenizer=tokenizer,
        callbacks=callbacks,  # try if doesnt work hashing all of checkpiint stuff above and also this callback line
        packing=False,
        # max_seq_length=1200
    )

    ###################################################

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
#import sys
#sys.path.append('/home/seatond/revision_project/code')
#from one import timer,finetune

if __name__ == '__main__':
    r=32;lora_alpha=64;lr = 0.8e-5;epochs = 1;warmup_ratio=0.13;wdecay=0.13
    target_modules=["q_proj", "v_proj","o_proj","k_proj","gate_proj","up_proj","down_proj","lm_head"]
    name = f"prompting_answer_rank{r}_lr{lr}_target{len(target_modules)}_epochs{epochs}_laplha{lora_alpha}_wuratio{warmup_ratio}_wdecay{wdecay}"
    trainer_obj = finetune(data,name, r, lora_alpha, lr, epochs, target_modules,warmup_ratio,wdecay)
# -----------------------------------------------------------------------------

# %%
hi = 1
if hi ==1 and __name__ == "__main__":
    trainer_obj.save_model()
