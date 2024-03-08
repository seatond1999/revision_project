# %% --------------------------------------------------------------------------
from flask import Flask, render_template, request
import pandas as pd
import os
from werkzeug.utils import secure_filename


import pandas as pd
from peft import PeftModel
from transformers import (
    AutoTokenizer,
    GenerationConfig,
    GPTQConfig,
    AutoModelForCausalLM,
)
import torch
from torch import nn
from openai import OpenAI
import openai
import os
import time
from PyPDF2 import PdfReader
import tensorboard
from auto_gptq import exllama_set_max_input_length
from app_class import app as appedy


# %% --------------------------------------------------------------------------
base_model_path = "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ"
# book = PdfReader(r"C:/MLE07/projects/edexcel_a_level_physics_student_book_1.pdf")
#book = PdfReader(r"../../business book for testing.pdf")

obj = appedy()
#load base model and adapters (if deployed could move)
# obj.get_base()
# obj.load_contclass_adapter()
# obj.load_fpclass_adapter()
# obj.load_splitcausal_adapter()
# obj.load_extrcontcausal_adapter()

# obj.get_contents()
# obj.extract_contents()
# obj.find_first_page()
# obj.find_first_page()


app = Flask(__name__)

# Set the path where uploaded files will be stored
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


#user uploads PDF,triggers functions to get corrected contents, diplsays contents to user.
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('upload.html', error='No file part')

        file = request.files['file']
        if file.filename == '':
            return render_template('upload.html', error='No selected file')

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            obj.book_filename = filename
            obj.book = PdfReader(file_path)
            # obj.get_contents()
            # obj.extract_contents()
            # obj.find_first_page()
            obj.contents = pd.DataFrame({'chapter_titles':[obj.book.pages[10].extract_text(),1],'testeroo':['huiefhiusehfuisef',23]})
            df = obj.contents
            return render_template('upload_success.html', filename=filename, column_names=df.columns)

        else:
            return render_template('upload.html', error='Invalid file format. Allowed formats: PDF')

    return render_template('upload.html', error=None)

@app.route('/contents', methods=['POST'])
def contents():
    filename = obj.book_filename
    # obj.get_contents()
    # obj.extract_contents()
    # obj.find_first_page()
    obj.contents = pd.DataFrame({'chapter_titles':[obj.book.pages[10].extract_text(),1],'testeroo':['huiefhiusehfuisef',23]})
    df = obj.contents
    print(df)
    return render_template('contents.html',filename=filename , tables=[df.to_html(classes='data', index=False)], column_names=df.columns)
    

#User has clicked on a chapter, triggers it to get breakdown and displays subtitles to user
@app.route('/update_df', methods=['POST'])
def update_df():
    filename = obj.book_filename
    clicked_value = request.form.get('clicked_value')
    print(clicked_value)
    df = obj.contents
    # Update DataFrame based on clicked value
    updated_data = {'Values': ['TESTINGINIGNING'+clicked_value]}
    obj.chapter_breakdown = pd.DataFrame(updated_data)
    updated_df = obj.chapter_breakdown
    
    return render_template('breakdown.html', filename=filename, tables=[df.to_html(classes='data', index=False), updated_df.to_html(classes='data', index=False)], column_names=df.columns)

#User has clicke don subtitle, now displaying just the question
@app.route('/qa_func', methods=['POST'])
def qa_func():
    filename = obj.book_filename
    updated_df = obj.chapter_breakdown
    df = obj.contents
    clicked_value = request.form.get('clickedValue')
    if not obj.question:
        obj.question = str(clicked_value) + 'hello user great choice'      
        obj.answer = 'wrong hahahaha'

    obj.qa = pd.DataFrame({'qa': [obj.question,obj.answer]})
    qa_df = obj.qa.iloc[0:1]

    return render_template('qa.html', filename=filename, tables=[df.to_html(classes='data', index=False), updated_df.to_html(classes='data', index=False),qa_df.to_html(classes='data', index=False)], column_names=df.columns)

#displaying q and a.
@app.route('/ans_func', methods=['POST'])
def ans_func():   
    filename = obj.book_filename 
    updated_df = obj.chapter_breakdown
    df = obj.contents
    obj.qa = pd.DataFrame({'qa': [obj.question,obj.answer]})
    qa_df = obj.qa

    
    return render_template('qa.html', filename=filename, tables=[df.to_html(classes='data', index=False), updated_df.to_html(classes='data', index=False),qa_df.to_html(classes='data', index=False)], column_names=df.columns)

@app.route('/reset', methods=['POST'])
def reset():
    filename = obj.book_filename
    obj.question = None
    obj.answer = None
    df = obj.contents
    return render_template('upload_success.html', filename=filename, tables=[df.to_html(classes='data', index=False)], column_names=df.columns)

if __name__ == '__main__':
    app.run()

# %%
