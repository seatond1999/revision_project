# %% --------------------------------------------------------------------------


#!pip install bitsandbytes
#!pip install -q -U git+https://github.com/huggingface/transformers.git
#!pip install -q -U git+https://github.com/huggingface/accelerate.git
#!pip install -q -U git+https://github.com/huggingface/peft.git
#!pip install -qqq torch
#!pip3 install auto-gptq --extra-index-url https://huggingface.github.io/autogptq-index/whl/cu118/
#!pip install optimum
#!pip install trl
#!pip install py7zr

# oneliner...
# pip install bitsandbytes git+https://github.com/huggingface/transformers.git git+https://github.com/huggingface/accelerate.git git+https://github.com/huggingface/peft.git torch auto-gptq --extra-index-url https://huggingface.github.io/autogptq-index/whl/cu118/ optimum trl py7zr
# pip3 install tensorboard scipy bitsandbytes git+https://github.com/huggingface/transformers.git git+https://github.com/huggingface/accelerate.git git+https://github.com/huggingface/peft.git auto-gptq optimum trl py7zr
# %% --------------------------------------------------------------------------
import pandas as pd
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GPTQConfig,
    TrainingArguments,
    TrainerCallback,
)
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer
import os
import tempfile
from huggingface_hub import login
from datetime import datetime as dt
import torch

# -----------------------------------------------------------------------------


## timer
def timer(func):
    def do(x):
        before = dt.now()
        perform = func(x)
        after = dt.now()
        print(after - before)
        return perform

    return do


# %%
token = "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"
login(token=token)

# preprocessing:


def prep_data():
    data = pd.read_csv(r"../ready_data.csv")
    # context had disctionary, only interested in the 'contexts' key

    # create data for asking Qs (aq)
    data_aq = pd.DataFrame(
        {
            "content": data.apply(
                lambda x: "###human: Ask me a questions about:\n"
                + x["full_context"]
                + "\n###Response:\n"
                + x["full_questions"],
                axis=1,
            )
        }
    )
    data_aq = Dataset.from_pandas(data_aq)
    data_aq = data_aq.train_test_split(test_size=0.016)
    return data_aq


@timer
def finetune(data):
    train_data = data["train"]
    test_data = data["test"]

    ##load model and tokenizer
    model_id = "TheBloke/Mistral-7B-v0.1-GPTQ"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    quantization_config_loading = GPTQConfig(
        bits=4, disable_exllama=True, tokenizer=tokenizer
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=quantization_config_loading, device_map="auto"
    )

    ##finetune:

    print(model)
    model.config.use_cache = False  # wont store intermediate states
    model.config.pretraining_tp = 1  # to replicate pre-training performance
    model.gradient_checkpointing_enable()  # compramise between forgetting activation states and remmebering them for backpropagation. Trade off computation time for GPU memory
    model = prepare_model_for_kbit_training(model)
    peft_config = LoraConfig(
        r=16,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"],
    )
    model = get_peft_model(model, peft_config)
    name = "mist_question_asking"
    training_arguments = TrainingArguments(
        output_dir=name,
        per_device_train_batch_size=8, #5 works
        gradient_accumulation_steps=1,
        optim="paged_adamw_32bit",
        learning_rate=3e-4,
        learning_rate=3e-4,
        lr_scheduler_type="cosine",
        save_strategy="epoch",  # so do we need the whole PeftSavingCallback function? maybe try withput and run the trainer(from last check=true)
        logging_steps=100,
        num_train_epochs=1,
        #max_steps=250,
        fp16=True,
        push_to_hub=True,
        report_to=["tensorboard"],
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
    # tmpdir = tempfile.TemporaryDirectory()
    # local_training_root = tmpdir.name

    # loc_checkpoint_path = os.path.join(local_training_root, name)
    # tensorboard_display_dir = f"{loc_checkpoint_path}/runs"

    # %load_ext tensorboard
    # %tensorboard --logdir f'{tensorboard_display_dir}'

    #########################################################

    #trainer.train(resume_from_checkpoint=True) #use if want to go from checkpoint
    trainer.train()
    #trainer.state.log_history()
    #trainer.save_model()
    #trainer.push_to_hub() #un hash when want to send final model to hub
    return trainer


# resume_from_checkpoint (str or bool, optional) — If a str, local path to a saved checkpoint as saved by a previous instance of Trainer. If a bool and equals True, load the last checkpoint in args.output_dir as saved by a previous instance of Trainer. If present, training will resume from the model/optimizer/scheduler states loaded here.


# %% --------------------------------------------------------------------------
# RUN!
if __name__ == "__main__":
    trainer_obj = finetune(prep_data())



# %% --------------------------------------------------------------------------
import os
os.getcwd()
# -----------------------------------------------------------------------------

# %%
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, GenerationConfig, GPTQConfig
import torch

#model_id_inf = "seatond/mist_question_asking_5e4LR_2epoch"

model_id_inf = r"/checkpoint-62/"

tokenizer = AutoTokenizer.from_pretrained(model_id_inf, use_fast=True)
# %%
