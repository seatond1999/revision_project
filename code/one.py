# %% --------------------------------------------------------------------------
#
# -----------------------------------------------------------------------------

import pandas as pd
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GPTQConfig,
    TrainingArguments,
    TrainerCallback,
    BitsAndBytesConfig,
    EarlyStoppingCallback
)
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer
import os
import tempfile
from huggingface_hub import login
from datetime import datetime as dt
import torch
import tensorboard
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
#token = "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"
#login(token=token)

# preprocessing:


def prep_data():
    data = pd.read_csv(r"../ready_data.csv")
    # context had disctionary, only interested in the 'contexts' key
    system = "You are a question-asking AI assisstant who will ask concise questions which can be answered using only the information provided by the user."
    # create data for asking Qs (aq)
    data_aq = pd.DataFrame(
        {
            "content": data.apply(
                lambda x: "<|im_start|>system "
                + f"{system}<|im_end|>"
                + "<|im_start|>user"
                + f" Information ###{x['full_context']}###<|im_end|>"
                + "<|im_start|>assisstant "
                + f"###{x['full_questions']}###<|im_end|>",
                axis=1,
            )
        }
    )

    data_aq = Dataset.from_pandas(data_aq)
    data_aq = data_aq.train_test_split(test_size=0.05)
    return data_aq


@timer
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
    tokenizer.add_tokens(["<|im_start|>"])
    tokenizer.add_special_tokens(dict(eos_token="<|im_end|>"))
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


# resume_from_checkpoint (str or bool, optional) — If a str, local path to a saved checkpoint as saved by a previous instance of Trainer. If a bool and equals True, load the last checkpoint in args.output_dir as saved by a previous instance of Trainer. If present, training will resume from the model/optimizer/scheduler states loaded here.


# %% --------------------------------------------------------------------------
# RUN!
# import os
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb=10'
if __name__ == "__main__":
    r=32;lora_alpha=64;lr = 1.5e-5;epochs = 1;warmup_ratio=0.14;wdecay=0.14
    target_modules=["q_proj", "v_proj","o_proj","k_proj","gate_proj","up_proj","down_proj","lm_head"]
    name = f"CHOSEN_rank{r}_lr{lr}_target{len(target_modules)}_epochs{epochs}_laplha{lora_alpha}_wuratio{warmup_ratio}_wdecay{wdecay}"
    trainer_obj = finetune(prep_data(),name, r, lora_alpha, lr, epochs, target_modules,warmup_ratio,wdecay)
# -----------------------------------------------------------------------------
# def finetune(data,r,lora_alpha,lr,epochs,target_modules,warmup_ratio,wdecay):

# %% --------------------------------------------------------------------------

# %% --------------------------------------------------------------------------
hi = 1
if hi ==1 and __name__ == "__main__":
    trainer_obj.save_model()
# -----------------------------------------------------------------------------

# %% --------------------------------------------------------------------------
hi = 0
if hi==1 and __name__ == "__main__": 
    prompt = "<|im_start|>system "+ f"You are a question-asking AI assisstant who will ask concise questions which can be answered using only the information provided by the user.<|im_end|>"+ "<|im_start|>user INFO"+ f"<|im_end|>"+ "<|im_start|>assisstant "+ f" QUESTIONS<|im_end|>"
    path = '/home/seatond/revision_project/code/NEW_everything_rank16_lr1.2e-05_target8_epochs1_laplha32_wuratio0.125_wdecay0.25/prompt.txt'
    with open(path, 'w') as file:
        file.write(prompt)
# %% --------------------------------------------------------------------------
hi = 1
if hi==1 and __name__ == "__main__":
    from huggingface_hub import HfApi

    hf_api = HfApi(
        endpoint="https://huggingface.co",  # Can be a Private Hub endpoint.
        token="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",  # Token is not persisted on the machine.
    )
    # token = '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'
    # login(token = token)
    # Upload all the content from the local folder to your remote Space.
    # By default, files are uploaded at the root of the repo
    hf_api.upload_folder(
        folder_path="/home/seatond/revision_project/code/CHOSEN_rank32_lr1.5e-05_target8_epochs1_laplha64_wuratio0.14_wdecay0.2",
        repo_id="seatond/CHOSEN_rank32_lr1.5e-05_target8_epochs1_laplha64_wuratio0.14_wdecay0.2",
        # repo_type="space",
    )
#max lengths of input training data is 1004

# %%
