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
    data = pd.read_json(r"testing_fp.json")
    df = pd.DataFrame(
        {
            "content": data.apply(
                lambda x: f"""<s>[INST] @@@ Instructions:
You are an assisstant who must classify whether a string from a page of a pdf book corresponds to the first page of a given chapter in that book.
You will be given the string and also given the chapter title.
You must reply with a single word which can be "yes" or "no"
You must reply "yes" if the string is the first page of the given chapter, and "no" if it is not the first page of the given chapter.

@@@ Example:
User: The given chapter title is ### The Chemistry Of Life ### and the string is: ### 'CHEMISTRY FOR BIOLOGISTS 61A.1 THE CHEMISTRY OF LIFE\nTHE CHEMISTRY OF WATER\nAll reactions in living cells take place in water. Without water, \nsubstances could not move around the body. Water is one of  the reactants in the process of  photosynthesis, on which almost all life depends (see fig E). Understanding the properties of  water will help you understand many key systems in living organisms. \nWater is also a major habitat – it supports more life than any other \npart of  the planet.\n▲ fig E  W ater is vital for life on Earth in many different ways – in a desert, \nthe smallest amount of water allows plants to grow.\nThe simple chemical formula of  water is H2O. This tells us that \ntwo atoms of  hydrogen are joined to one atom of  oxygen to make \nup each water molecule. However, because the electrons are held closer to the oxygen atom than to the hydrogen atoms, water is a polar molecule (see fig F).\n104.5°Oδ2\nHδ1Hδ1\n▲ fig F  A model of a w ater molecule showing dipoles.\nOne major effect of  this polarity is that water molecules form hydrogen  bonds. The slightly negative oxygen atom of  one water \nmolecule will attract the slightly positive hydrogen atoms of  other water molecules in a weak electrostatic attraction called a hydrogen \nbond. Each individual hydrogen bond is weak but there are many of  them so the molecules of  water ‘stick together’ more than you might expect (see fig G). Water has relatively high melting and boiling points compared with other substances that have molecules of  a similar size because it takes a lot of  energy to break all the hydrogen bonds that hold the molecules together. Hydrogen bonds are important in protein structure (see Sections 1A.5 and 2B.1) \nand in the structure and functioning of  DNA (see Section 2B.3).Oδ2\nOδ2\nHδ1Hδ1Hδ1Hδ1\nHδ1Oδ2Oδ2\nOδ2Hδ1\nHδ1\nHδ1Hδ1Hδ1\n▲ fig G  Hydr ogen bonding in water molecules, based on attraction \nbetween positive and negative dipoles.\nTHE IMPORTANCE OF WATER\nThe properties of  water make it very important in biological \nsystems for many reasons.\n •W ater is a polar solvent. Because it is a polar molecule, many ionic \nsubstances like sodium chloride will dissolve in it (see fig H).  \nMany covalently bonded substances are also polar and will dissolve in water, but often do not dissolve in other covalently bonded solvents such as ethanol. Water also carries other substances, such as starch. As a result, most of  the chemical reactions within cells occur in water (in aqueous solution).\nsodium and chloride ionsin solution in water\nsalt and water mixed\nsodium chloride\nNaClionic bond sodium ionchlorideionδ1 chargeson hydrogenin water areattracted tonegativechloride ion\nδ2\n charges\non oxygenin water ar\ne\nattracted tothe positivesodium ionH\nH\nHHH\nH HO\nOO\nCl2\nCl2\nCl2Cl2\nCl2\nNa1OH\nCl2\nNa1Na1\nCl2Cl2\nCl2\nCl2Cl2\nNa1Na1Na1 Na1\nNa1Na1Na1\n▲ fig H  A model of sodium chloride dissolving in water as a result of the \ninteractions between the charges on sodium and chloride ions and the dipoles of the water molecules.\nUncorrected proof, all content subject to change at publisher discretion. Not for resale, circulation or distribution in whole or in part. ©Pearson 2018' ###
Assisstant: "Yes Yes Yes Yes Yes"

@@@ Example:
User: The given chapter title is ### Preparing For Your Exams ### and the string is: ### 'xASSESSMENT OVERVIEW\nPAPER / UNIT 1PERCENTAGE \nOF IASPERCENTAGE OF IALMARK TIME AVAILABILITY\nMOLECULES, DIET, TRANSPORT AND \nHEALTH \nWritten examination\nPaper code \nWBI11/01\nExternally set and marked by \nPearson Edexcel\nSingle tier of entry40% 20% 80 1 hour  \n30 minutesJanuary, June and October\nFirst assessment : January 2019\nPAPER / UNIT 2PERCENTAGE \nOF IASPERCENTAGE OF IALMARK TIME AVAILABILITY\nCELLS, DEVELOPMENT, BIODIVERSITY \nAND CONSERVATION\nWritten examination\nPaper code \nWBI12/01\nExternally set and marked by \nPearson Edexcel\nSingle tier of entry40% 20% 80 1 hour  \n30 minutesJanuary, June and October\nFirst assessment : June 2019\nPAPER / UNIT 3PERCENTAGE \nOF IASPERCENTAGE OF IALMARK TIME AVAILABILITY\nPRACTICAL SKILLS IN BIOLOGY 1   \nWritten examination\nPaper code \nWBI13/01\nExternally set and marked by \nPearson Edexcel\nSingle tier of entry20% 10% 50 1 hour  \n20 minutesJanuary, June and October\nFirst assessment : June 2019ASSESSMENT OVERVIEW\nThe following tables give an overview of the assessment for Pearson Edexcel International Advanced Subsidiary course \nin Biology. You should study this information closely to help ensure that you are fully prepared for this course and know exactly what to expect in each part of the examination. More information about this qualification, and about the question types in the different papers, can be found on page 302 of this book.\nUncorrected proof, all content subject to change at publisher discretion. Not for resale, circulation or distribution in whole or in part. ©Pearson 2018' ###
Assisstant: "No No No No No"

@@@ Question:
User: The given chapter title: ### {x['chapter_title']} ### This is the string: ### {x['page']} ### [/INST]
Assisstant: {x['label']}""",
                axis=1,
            ),
            'label': data.apply(lambda x: 0 if x['label']=='No' else 1,axis=1)
        }
    )

    #stratified sampling
    df = df.sample(frac=1)
    df.reset_index(inplace=True)
    df.drop(columns='index',inplace=True)
    split = 0.15

    test = Dataset.from_pandas(pd.concat([df[df['label']==0][0:int(len(df[df['label']==0])*split)] ,  df[df['label']==1][0:int(len(df[df['label']==1])*split)]]))
    train = Dataset.from_pandas(pd.concat([df[df['label']==0][int(len(df[df['label']==0])*split):] ,  df[df['label']==1][int(len(df[df['label']==1])*split):]]))
    
    #return data_aq
    return train,test


@timer
def finetune(data, r, lora_alpha, lr, epochs, target_modules,batch_s,gradacc):
    token = "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"
    login(token=token)
    train_data = data[0]
    test_data = data[1]

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
    name = f"TESTfirstpage_rank{r}_lr{lr}_target{len(target_modules)}_epochs{epochs}_laplha{lora_alpha}_batch{batch_s}_gradacc{gradacc}" #_wuratio{warmup_ratio}_wdecay{wdecay    
    training_arguments = TrainingArguments(
        output_dir=name,
        per_device_train_batch_size=batch_s,  # 5 works
        per_device_eval_batch_size=batch_s,
        gradient_accumulation_steps=gradacc,
        optim="paged_adamw_32bit",
        learning_rate=lr,
        lr_scheduler_type="cosine",
        save_strategy="steps",  # so do we need the whole PeftSavingCallback function? maybe try withput and run the trainer(from last check=true)
        logging_steps=12,
        #save_steps=60,
        num_train_epochs=epochs,
        max_steps=49,
        fp16=False,
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
        max_seq_length=3700
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
    data = prep_data()
    trainer_obj = finetune(data, 16, 32, 2.2e-5, 1, ["q_proj", "v_proj","o_proj","k_proj"],1,4)

#,"gate_proj"
#,"gate_proj","up_proj","down_proj"
# -----------------------------------------------------------------------------
# def finetune(data,r,lora_alpha,lr,epochs,target_modules,batch,,warmup_ratio,wdecay):

# %% --------------------------------------------------------------------------
trainer_obj.save_model()
# -----------------------------------------------------------------------------

# %% --------------------------------------------------------------------------

# %% --------------------------------------------------------------------------
#lens = []
#for i in prep_data()[0]['content']:
#    lens.append(len(trainer_obj(i)['input_ids']))


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
        folder_path="/home/seatond/revision_project/revamp/TESTfirstpage_rank64_lr2.3e-05_target7_epochs1_laplha128_batch1_gradacc4",
        repo_id="seatond/multi_yes_short",
        # repo_type="space",
    )
# %%
