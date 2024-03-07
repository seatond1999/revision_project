
# %% --------------------------------------------------------------------------

# import pandas as pd
# from peft import PeftModel
# from transformers import (
#     AutoTokenizer,
#     GenerationConfig,
#     GPTQConfig,
#     AutoModelForCausalLM,
# )
# import torch
# from torch import nn
# from openai import OpenAI
# import openai
# import os
# import time
# from PyPDF2 import PdfReader
# import tensorboard
# from auto_gptq import exllama_set_max_input_length
# %%
# %% --------------------------------------------------------------------------
#remember for this script need to havemodified mistral_modelling.py file
#have to load book and set as attirbute
class app:
    def __init__(self,base_model_path):
        self.book = None
        self.base_model_path = base_model_path
        self.model = None
        self.tokenizer = None
        self.fpclass_head = None
        self.contclass_head = None
        self.causal_head = None
        self.page_lag = None
        self.contents_first = None
        self.contents_last = None
        self.contents = None #make pd dataframe but get last contents page first lol.
        self.chapter_breakdown = None
        self.qa = None
        #for flask
        self.book_filename = None
        self.question = None
        self.answer = None

    def load_base_model(base_model):
        base_path = base_model  # input: base model

        tokenizer = AutoTokenizer.from_pretrained(base_path, use_fast=False)

        quantization_config_loading = GPTQConfig(
        bits=4, #this is usually 4!
        disable_exllama=False,
        tokenizer=tokenizer,
    )

        model = AutoModelForCausalLM.from_pretrained(
        base_path,
        quantization_config=quantization_config_loading,
        device_map="auto",
        torch_dtype=torch.float16,
    )
        #model = exllama_set_max_input_length(model, max_input_length=33400)

        tokenizer.pad_token = "<unk>"
        tokenizer.padding_side = "right"
        model.resize_token_embeddings(len(tokenizer))
        model.config.eos_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id

        #model.config.use_return_dict = True #so changes to source code work.

        return tokenizer,model

    def get_base(self):
        self.tokenizer, self.model = self.load_base_model(self.base_model_path)
        self.causal_head = self.model.lm_head #no finetuning was done to causal head, unlike classificaiton head, so can take the base model LM_head for base and causal.


    def load_fpclass_adapter(self):
        self.model.num_labels = 2
        self.model.config.num_labels = 2 #model attributes needed for classificaiton forward pass
        self.model.score = nn.Linear(self.model.config.hidden_size, self.model.config.num_labels, bias=False,dtype=torch.float16) #bias is false as finetuned with and without and made no diff so saves memory
        self.model.load_adapter(peft_model_id=r'../adapters/firstpage',adapter_name='ident_fistpage_adapter')
        self.fpclass_head = self.model.score
        return None

    def load_contclass_adapter(self):
        self.model.num_labels = 2
        self.model.config.num_labels = 2 #model attributes needed for classificaiton forward pass
        self.model.score = nn.Linear(self.model.config.hidden_size, self.model.config.num_labels, bias=False,dtype=torch.float16) #bias is false as finetuned with and without and made no diff so saves memory
        self.model.load_adapter(peft_model_id=r'seatond/identcontents_rank16_lr2.2e-05_target3_39steps_laplha32_batch1_gradacc4',adapter_name='ident_cont_adapter')
        self.contclass_head = self.model.score
        return None

    def load_splitcausal_adapter(self):
        self.model.lm_head = self.causal_head
        self.model.load_adapter(peft_model_id='seatond/revamped_rank64_batch4',adapter_name='splitcausal_adapter') #should make the adapter paths etc dynamic

    def load_extrcontcausal_adapter(self):
        self.model.lm_head = self.causal_head
        self.model.load_adapter(peft_model_id='seatond/EXTRACTION_rank32_lr2.2e-05_target7_epochs1.7_laplha64_batch1_gradacc4',adapter_name='extrcontcausal_adapter')

    def change_adapter(self,desired): #desired should be classification,causal,base
        if desired == 'fpclass':
            self.model.lm_head = None #replaced LM_head with score which maps to our 2 classes
            self.model.config.lm_head = None  #remove mpapping to vocab size
            self.model.conditioner = 'classification' #added if statement in source code to direct to correct type of forward pass for classification.
            self.model.score = self.fpclass_head #sets the 2 class head
            self.model.enable_adapters()
            self.model.set_adapter('ident_fistpage_adapter')
        if desired == 'contclass':
            self.model.lm_head = None #replaced LM_head with score which maps to our 2 classes
            self.model.config.lm_head = None  #remove mpapping to vocab size
            self.model.conditioner = 'classification' #added if statement in source code to direct to correct type of forward pass for classification.
            self.model.score = self.contclass_head #sets the 2 class head
            self.model.enable_adapters()
            self.model.set_adapter('ident_cont_adapter')
        if desired == 'splitcausal':
            self.model.score = None
            self.model.config.score = None
            self.model.conditioner = 'causal'
            self.model.lm_head = self.causal_head
            self.model.enable_adapters()
            self.model.set_adapter('splitcausal_adapter')
        if desired == 'extrcontcausal':
            self.model.score = None
            self.model.config.score = None
            self.model.conditioner = 'causal'
            self.model.lm_head = self.causal_head
            self.model.enable_adapters()
            self.model.set_adapter('extrcontcausal_adapter')
        if desired == 'base':
            self.model.score = None
            self.model.config.score = None
            self.model.conditioner = 'causal'
            self.model.lm_head = self.causal_head #looks similar to causal as base is a mistralforcausal LM
            self.model.disable_adapters()


    def get_contents(self):
            self.change_adapter('contclass')
            from auto_gptq import exllama_set_max_input_length
            obj.model = exllama_set_max_input_length(obj.model, max_input_length=30761)
            prompt = """<s>[INST] @@@ Instructions:
        It is your task to classify whether a string corresponds to the contents page of a pdf book.
        A contents page includes chapter titles and page numbers.
        Only reply with the words "Yes" or "No"
        You must reply "yes" if the string is from the contents page, and "no" if it is not the contents page.

        @@@ Question:
        This is the string: ### """
            generation_config = GenerationConfig(
                do_sample=True,
                top_p=0.95, top_k=40,
                temperature=0.7,
                max_new_tokens=150,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )

            for i in range(0,30):
                input_ids = self.tokenizer(prompt+self.book.pages[i].extract_text()+' ### [/INST]',return_tensors='pt').input_ids.cuda()
                input_ids = input_ids.to("cuda:0") #making sure not on CPU as causes error
                self.model = self.model.to("cuda:0")
                with torch.no_grad(): #no grad as dont want to change any weights just doing an inference
                    predicted_class = torch.argmax(
                        self.model(input_ids).logits
                        ).item() # the inner bit of argmax just gets the logits, we then take max value to get predicted class
                    #no need for softmax as only 2 classes.
                if predicted_class == 1:
                    self.contents_first = i
                    break
            for i in range(self.contents_first,30):
                input_ids = self.tokenizer(prompt+self.book.pages[i].extract_text()+' ### [/INST]',return_tensors='pt').input_ids.cuda()
                input_ids = input_ids.to("cuda:0") #making sure not on CPU as causes error
                self.model = self.model.to("cuda:0")
                with torch.no_grad(): #no grad as dont want to change any weights just doing an inference
                    predicted_class = torch.argmax(
                        self.model(input_ids).logits
                        ).item()
                if predicted_class == 0:
                    self.contents_last = i-1
                    break
            #could make 2 things above a funciton to reduce repetition?
            return 'found contents page .. hopefully'

    def extract_contents(self):
        self.change_adapter('extrcontcausal')
        contents_pages = "" #if method too long can feed separate prompts to LLM for each page of contents page.
        for i in range(self.contents_first,self.contents_last+1):
            contents_pages += self.book.pages[i].extract_text()

        prompt = """<s>[INST] @@@ Instructions:
It is your task to extract the chapters and corresponding page numbers from a string which was created from the contents page of a pdf book.
You must return a list of the chapters and page numbers.
Put each chapter and its page number on its own line, and separate chapters titles from page numbers with a "---".
You will be penalised for separating chapters with anything that is not "---"
For example the first 2 chapters of a contents page should be in the following format: "chapter 1 title --- chapter 1 page number \n chapter 2 title --- chapter 2 page number"

@@@ Question:
string which was created from the contents page of a pdf book: ### """

        input_ids = self.tokenizer(prompt+contents_pages+' ### [/INST]',return_tensors='pt').input_ids.cuda()
        output = self.tokenizer.decode((self.model.generate(inputs=input_ids, temperature=0.07, do_sample=True, top_p=0.35, top_k=5, max_new_tokens=2000))[0])

        contents_list = [i.split('---') for i in output.split('\n')]
        contents_df = pd.DataFrame(contents_list,columns=['chapter_titles','page_number'])

        mask = contents_df['page_number'].notna()
        contents_df = contents_df[mask]
        contents_df['page_number'] = contents_df['page_number'].apply(lambda x: x.replace(' ',''))
        mask = contents_df['page_number'].apply(lambda x: str(x).isnumeric())
        contents_df = contents_df[mask]
        contents_df.reset_index(inplace=True)
        contents_df.drop(columns=['index'],inplace=True)
        contents_df['page_number'] = contents_df['page_number'].apply(lambda x: int(x))
        contents_df = contents_df.sort_values(by='page_number', ascending=True)
        contents_df.reset_index(inplace = True)
        contents_df.drop(columns = ['index'],inplace = True)
        self.contents = contents_df

        return contents_df

    def ffp_checker(self): #check function using page lag on rando pages and checking goes from no to yes.
        check_start = self.contents['page_number'][3] + self.page_lag - 1
        check_end = self.contents['page_number'][3] + self.page_lag + 1
        for i in range(check_start,check_end): #gonna loop through pages until find first page of chapter
                  #each iteration will do a forward pass through the model with adapters with page i book
                  prompt = f"""<s>[INST] This is a string from a page of a pdf book: ### {self.book.pages[i].extract_text()} ###
                  Is it true or false that this page belongs to a chapter called: ### {self.contents['chapter_titles'][3]} ###? [/INST]"""
                  input_ids = self.tokenizer(prompt,return_tensors='pt').input_ids.to("cuda:0")
                  input_ids = input_ids.to("cuda:0") #making sure not on CPU as causes error

                  with torch.no_grad(): #no grad as dont want to change any weights just doing an inference
                      predicted_class = torch.argmax(
                          self.model(input_ids).logits
                          ).item() # the inner bit of argmax just gets the logits, we then take max value to get predicted class
                      #no need for softmax as only 2 classes

                  if i == check_start and predicted_class == 1:
                      return 0
                  elif i !=check_start and predicted_class == 0:
                      return 0
        return 1

    def ffp_finder(self,pos):
            find_start = self.contents['page_number'][pos] if pos!=0 else self.contents_first+1
            find_end = find_start + 30
            for i in range(find_start,find_end): #gonna loop through pages until find first page of chapter
                #each iteration will do a forward pass through the model with adapters with page i book
                prompt = f"""<s>[INST] This is a string from a page of a pdf book: ### {self.book.pages[i].extract_text()} ###
                Is it true or false that this page belongs to a chapter called: ### {self.contents['chapter_titles'][pos]} ###? [/INST]"""
                input_ids = self.tokenizer(prompt,return_tensors='pt').input_ids.to("cuda:0")
                input_ids = input_ids.to("cuda:0") #making sure not on CPU as causes error
                self.model = self.model.to("cuda:0")
                with torch.no_grad(): #no grad as dont want to change any weights just doing an inference
                    predicted_class = torch.argmax(
                        self.model(input_ids).logits
                        ).item() # the inner bit of argmax just gets the logits, we then take max value to get predicted class
                    #no need for softmax as only 2 classes

                if predicted_class == 1:
                    self.page_lag = i - (self.contents['page_number'][pos]) #getting diff betweeen contents page of title and actual page
                    break
                else:
                    None

    def find_first_page(self):
            self.change_adapter('fpclass')
            for i in range(0,3):
                print(i)
                self.ffp_finder(i)
                if self.ffp_checker() == 1:
                    self.contents['corrected_page_number'] = self.contents['page_number'].apply(lambda x: x + self.page_lag)
                    return 'found contents page .. hopefully'
                else:
                    None
            self.page_lag = 'error finding pages.'
            return 'there has been an error.'

###########################################retreival part done now
## add page number to the table?
    def split_chosen_chapter(self,chosen_chapter): #if results come out bad can do the thing where chop off end of page and put into next one so dont get crap endings.
        self.change_adapter('splitcausal')
        self.model = self.model.to("cuda:0")
        chosen_chapter_index = self.contents[self.contents['chapter_titles'] == chosen_chapter].index[0] #gets row of chosen chapter in contents df
        responses = ""
        for i in range(self.contents.iloc[chosen_chapter_index,2],self.contents.iloc[chosen_chapter_index,2]+2): #between the page numbers of chosen chapter and next chapter
            prompt = f"""[INST]You must split text up into subsections and add informative titles for each subsection.
Each subsection must be in paragraph form and all information should be included from the original text.
You will be penalized for removing information from the original text.
Mark each title you create by adding the symbols "@@@" before each title and placing the title on its own line.
An example subsection format is "@@@title \n content", where you should add the subsection title and content.
This is the text:
### {self.book.pages[i].extract_text()} ### [/INST]
Output: """

            input_ids = self.tokenizer(prompt,return_tensors='pt').input_ids.to("cuda:0")
            input_ids = input_ids.to("cuda:0")

            generation_config = GenerationConfig(
              do_sample=True,
              top_p=0.95, top_k=40,
              temperature=0.7,
              max_new_tokens=150000,
              eos_token_id=self.tokenizer.eos_token_id,
              pad_token_id=self.tokenizer.pad_token_id,
            )


            out = self.model.generate(inputs=input_ids, generation_config=generation_config)
            decoded_output = self.tokenizer.decode(out[0], skip_special_tokens=True)
            decoded_output = decoded_output[
                decoded_output.find("[/INST]")
                + len("[/INST]"):
            ]

            responses += decoded_output


        responses = responses.split('@@@')[1:]
        self.chapter_breakdown = pd.DataFrame({
            'subtitle':list(map(lambda x: x[:x.find('\n')],responses)),
            'text':list(map(lambda x: x[x.find('\n')+2:],responses))})
        return 'done'
        
    # this function will ask the base mistral model a question about the subtitle section which the user has chosen
    # for now have chosen sub as the indice in the chapter breakdown table
    def question_answer(self,chosen_sub):
        self.change_adapter('base')
        self.model = self.model.to("cuda:0")

        prompt = f"""[INST]You are a helpful revision assisstant who asks the user a question about some information which you are given and provide the answer and a concise explanation.
        Your questions must be based on the provided information only.
        Your answer and explanation must include knowledge from the provided information only.
        You will be rewarded for making the explanation a single sentence.
        You must format your response in the following way:
        ---
        Question  (replace with actual question)
        @@@
        Answer (replace with actual Answer)
        @@@
        Explanation (replace with actual Explanation)
        ---
        You will be penalised for doing a different format to the one shown above.
        This is the provided information:
        ### {self.chapter_breakdown['text'][chosen_sub]} ### [/INST]
        Output: """

        input_ids = self.tokenizer(prompt,return_tensors='pt').input_ids.to("cuda:0")
        input_ids = input_ids.to("cuda:0")

        generation_config = GenerationConfig(
          do_sample=True,
          top_p=0.95, top_k=40,
          temperature=0.7,
          max_new_tokens=150000,
          eos_token_id=self.tokenizer.eos_token_id,
          pad_token_id=self.tokenizer.pad_token_id,
        )


        out = self.model.generate(inputs=input_ids, generation_config=generation_config)
        decoded_output = self.tokenizer.decode(out[0], skip_special_tokens=True)
        decoded_output = decoded_output[
            decoded_output.find("[/INST]")
            + len("[/INST]"):
        ]
        decoded_output = decoded_output[
            decoded_output.find("---")
            + len("---"):
        ]
        decoded_output = decoded_output.split('@@@')
        qa_df = pd.DataFrame({'qa':decoded_output})

        self.qa = qa_df
        return 'done'


# %% --------------------------------------------------------------------------
# base_model_path = "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ"
# book = PdfReader(r"../../edexcel_a_level_physics_student_book_1.pdf")
# #book = PdfReader(r"../../business book for testing.pdf")

# obj = app(base_model_path,book)
# obj.get_base()

# # %% --------------------------------------------------------------------------
# load_base_model("TheBloke/Mistral-7B-Instruct-v0.2-GPTQ")
# # -----------------------------------------------------------------------------
