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


# %% --------------------------------------------------------------------------
from app_class import app as appedy
# book = PdfReader(r"C:/MLE07/projects/edexcel_a_level_physics_student_book_1.pdf")
#book = PdfReader(r"../../business book for testing.pdf")

obj = appedy()
#load base model and adapters (if deployed could move)
# obj.get_base()
# obj.load_contclass_adapter()
# obj.load_fpclass_adapter()
# obj.load_splitcausal_adapter()
# obj.load_extrcontcausal_adapter()


# %% --------------------------------------------------------------------------
# -----------------------------------------------------------------------------
app = Flask(__name__)

# Set the path where uploaded files will be stored
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

#user uploads PDF,triggers functions to get corrected contents, diplsays contents to user.
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    obj.contents = None
    obj.chapter_breakdown = None
    obj.questions = None
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
            return render_template('upload_success.html', filename=filename)

        else:
            return render_template('upload.html', error='Invalid file format. Allowed formats: PDF')

    return render_template('upload.html', error=None)

@app.route('/contents', methods=['POST'])
def contents():
    filename = obj.book_filename
    if obj.contents is None:
        print('starting stuff')
        #####################
        obj.get_contents()
        obj.extract_contents()
        obj.find_first_page()
        # obj.contents_first = loaded_contents_first
        # obj.contents_last = loaded_contents_last
        # obj.contents = loaded_contents
        #####################
        print('ending stuff')
    df = obj.contents
    print(df)
    return render_template('contents.html',filename=filename , tables=[df.to_html(classes='data', index=False)], column_names=df.columns)
    

#User has clicked on a chapter, triggers it to get breakdown and displays subtitles to user
@app.route('/update_df', methods=['POST'])
def update_df():
    print('starting dupate_df')
    clicked_chapter = request.form.get('chapter_title')
    if obj.chapter_breakdown is None:
        ###################
        obj.split_chosen_chapter(clicked_chapter+' ')
        print(clicked_chapter)

        # obj.chapter_breakdown = loaded_chapter_breakdown
        ################################
    updated_df = obj.chapter_breakdown.drop(columns=['text'])
    
    return render_template('breakdown.html', tables=[updated_df.to_html(classes='data', index=False)],title=clicked_chapter)

@app.route('/expand_sub', methods=['POST'])
def expand_sub():
    updated_df = obj.chapter_breakdown.drop(columns=['text'])
    clicked_subtitle = request.form.get('clickedValue')
    clicked_chapter = request.form.get('chapter_title')
    print(clicked_subtitle)
    index = obj.chapter_breakdown[obj.chapter_breakdown['subtitle']==clicked_subtitle].index[0]
    expanded_sub = obj.chapter_breakdown['text'][index]
    return render_template('expand_sub.html',subtitle=clicked_subtitle,chapter=clicked_chapter,expanded_sub = expanded_sub,tables=[updated_df.to_html(classes='data', index=False)],title=clicked_chapter)


#User has clicke don subtitle, now displaying just the question
@app.route('/qa_func', methods=['POST'])
def qa_func():
    clicked_subtitle = request.form.get('subtitle')
    clicked_chapter = request.form.get('chapter_title')
    if not obj.questions:
        print(clicked_subtitle)
        #############################
        obj.question(clicked_subtitle)
        obj.answer(clicked_subtitle)
        # obj.questions = loaded_questions
        # obj.answers = loaded_answers
        #############################
    questions = obj.questions

    return render_template('questions.html', questions = questions,subtitle=clicked_subtitle,title = clicked_chapter)
    

#displaying q and a.
@app.route('/ans_func', methods=['POST'])
def ans_func():   
    clicked_subtitle = request.form.get('subtitle')
    clicked_chapter = request.form.get('chapter_title')
    print('currrrrr' + clicked_chapter)
    questions = obj.questions
    answers = obj.answers

    return render_template('answers.html', questions = questions,subtitle=clicked_subtitle,answers=answers,title = clicked_chapter)

@app.route('/change_chapter', methods=['GET'])
def change_chapter():
    df = obj.contents
    return render_template('contents.html', tables=[df.to_html(classes='data', index=False)], column_names=df.columns)

@app.route('/change_subtitle')
def change_subtitle():
    clicked_chapter = request.args.get('chapter_title')
    updated_df = obj.chapter_breakdown.drop(columns=['text'])
    obj.questions = None
    return render_template('breakdown.html', tables=[updated_df.to_html(classes='data', index=False)],title=clicked_chapter)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

# %%

