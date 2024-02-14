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
token = "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"
login(token=token)

def prep_data():
    data = pd.read_csv(r"../data/prepared_data.csv")
    data = data[data['extraction'] != 'IGNORE']
    data_aq = pd.DataFrame(
        {
            "content": data.apply(
                lambda x: f"""<s>[INST] @@@ Instructions:
It is your task to extract the chapters and corresponding page numbers from a string which was created from the contents page of a pdf book.
You must return a list of the chapters and page numbers.
Put each chapter and its page number on its own line, and separate chapters titles from page numbers with a "---".
You will be penalised for separating chapters with anything that is not "---"
For example the first 2 chapters of a contents page should be in the following format: "chapter 1 title --- chapter 1 page number \n chapter 2 title --- chapter 2 page number"

@@@ Question:
string which was created from the contents page of a pdf book: ### {x['contents_page']} ### [/INST]

Output: {x['extraction']}""",
                axis=1,
            )
        }
    )
    data_aq = Dataset.from_pandas(data_aq)
    data_aq = data_aq.train_test_split(test_size=0.1)
    return data_aq


@timer
def finetune(data, r, lora_alpha, lr, epochs, target_modules,batch_s,gradacc):
    token = "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"
    login(token=token)
    train_data = data["train"]
    test_data = data["test"]

    ##load model and tokenizer
    model_id = "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ"

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

    tokenizer.pad_token = "<unk>"
    tokenizer.padding_side = "right"
    model.resize_token_embeddings(len(tokenizer))
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id


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
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    model = get_peft_model(model, peft_config)
    print("trainable parameters:", model.print_trainable_parameters())
    ##
    name = f"EXTRACTION_rank{r}_lr{lr}_target{len(target_modules)}_epochs{epochs}_laplha{lora_alpha}_batch{batch_s}_gradacc{gradacc}" #_wuratio{warmup_ratio}_wdecay{wdecay    
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
        # max_steps=250,
        fp16=False, #only set it to False for today's fietune (01/02/2024)
        push_to_hub=False,
        report_to=["tensorboard"],
        group_by_length=True,
        ddp_find_unused_parameters=False,
        do_eval=True,  ## for eval:
        evaluation_strategy="steps",
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
        max_seq_length=3500 # did by tokenizing all dtaa set and getting max length
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
    trainer_obj = finetune(prep_data(), 32, 64, 2.2e-5, 2, ["q_proj", "v_proj","o_proj","k_proj","up_proj","down_proj","gate_proj"],1,4)

#,"gate_proj"
#,"gate_proj","up_proj","down_proj"
# -----------------------------------------------------------------------------
# def finetune(data,r,lora_alpha,lr,epochs,target_modules,batch,,warmup_ratio,wdecay):

# %% --------------------------------------------------------------------------
trainer_obj.save_model()
# -----------------------------------------------------------------------------

# %% --------------------------------------------------------------------------

# %% --------------------------------------------------------------------------


# %%
#see max sequence length of input
hi = 0
if hi == 1:
    from transformers import AutoTokenizer
    model_id = "TheBloke/Mistral-7B-v0.1-GPTQ"

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, use_fast=False, add_eos_token=True)
    
    lens = []
    for i in prep_data()['train']['content']:
        lens.append(len(tokenizer(i)['input_ids']))

    print(max(lens))



# %%

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
        folder_path="/home/seatond/revision_project/revamp/stage_2/code/DADSADASDASDASDASDASDv2_gp4_rank64_lr2.2e-05_target7_epochs2_laplha128_batch1_gradacc4",
        repo_id="seatond/chosen_stage2_exllamatrue",
        # repo_type="space",
    )
# %%
