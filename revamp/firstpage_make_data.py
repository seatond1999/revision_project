# %% --------------------------------------------------------------------------
from PyPDF2 import PdfReader
import pandas as pd

# -----------------------------------------------------------------------------

# %% --------------------------------------------------------------------------
# aqa biology
[]
yes = [
    #yes.... add 5
    2,
    6,#Lipids
    8, #Proteins
    10,#Enzyme Action
    12,#Factors Affecting Enzyme Activity
    14,
    16,
    18,
    20,
    22,
    23,
    24,
    28,
    30,
    32,
    34,
    36,
    38,
    40,
    42,
    44,
    46,
    48,
    50,
    52,
    54,
    56,
    58,
    60,
    62,
    64,
    146,
    148,
    152,
    154,
    156,
    159,
    162,
    164,
    166,
    168,
    170,
    172,
    174,
    176,
    178,
    180,
    182,
    184,
    186,
    190,
    193,
    195,
    196,
    199,
    201,
    204,
    206,
    208,
    210,
    213,
    215,
    217,
]
yes_titles = [
    'Carbohydrates',
    'Lipids',
    'Proteins',
    'Enzyme Action',
    'Factors Affecting Enzyme Activity',
    'Enzyme-Controlled Reactions',
    'DNA and RNA',
    'DNA Replication',
    'Water',
    'ATP',
    'Inorganic Ions',
    'Eukaryotic Cells and Organelles',
    'Prokaryotic Cells and Viruses',
    'Analysis of Cell Components',
    'Cell Division - Mitosis',
    'Cell Division - Investigating Mitosis',
    'Cell Membrane Structure',
    'Exchange Across Cell Membranes - Diffusion',
    'Exchange Across Cell Membranes - Osmosis',
    'Exchange Across Cell Membranes - Active Transport',
    'The Immune System',
    'Immunity and Vaccines',
    'Antibodies in Medicine',
    'Interpreting Vaccine and Antibody Data',
    'HIV and Viruses',
    'Size and Surface Area',
    'Gas Exchange',
    'Gas Exchange in Humans',
    'The Effects of Lung Disease',
    'Interpreting Lung Disease Data',
    'Dissecting Gas Exchange Systems',
    'Homeostasis Basics',
    'Control of Blood Glucose Concentration',
    'The Kidneys',
    'Controlling Blood Water Potential',
    'Inheritance',
    'Linkage and Epistasis',
    'The Chi-Squared Test',
    'The Hardy-Weinberg Principle',
    'Variation and Selection',
    'Speciation and Genetic Drift',
    'Ecosystems',
    'Variation in Population Size',
    'Investigating Populations',
    'Succession',
    'Conservation',
    'Mutations',
    'Cancer',
    'Interpreting Data on Cancer',
    'Stem Cells',
    'Regulation of Transcription and Translation',
    'Epigenetic Control of Gene Expression',
    'Evaluating Data on Phenotypes',
    'Genome Projects and Making DNA Fragments',
    'Amplifying DNA Fragments',
    'Using Recombinant DNA Technology',
    'Gene Probes and Medical Diagnosis',
    'Gentic Fingerprinting',
    'Planning an Experiment',
    'Processing and Presenting Data',
    'Drawing Conclusions and Evaluating',
    'How To Do Well in Your Exams',
    'Answers'
]

#just minus 1:
no = [
    2,
    2,
    4,
    13,
    14,
    16,
    18,
    20,
    22,
    24,
    26,
    28,
    29,
    30,
    34,
    36,
    38,
    40,
    42,
    44,
    46,
    48,
    50,
    52,
    54,
    56,
    58,
    60,
    62,
    64,
    66,
    68,
    70,
    72,




]

no_titles = [
    'Carbohydrates',
    'How To Do Well in Your Exams',
    'Drawing Conclusions and Evaluating',
    'Carbohydrates',
    'Lipids',
    'Proteins',
    'Enzyme Action',
    'Factors Affecting Enzyme Activity',
    'Enzyme-Controlled Reactions',
    'DNA and RNA',
    'DNA Replication',
    'Water',
    'ATP',
    'Inorganic Ions',
    'Eukaryotic Cells and Organelles',
    'Prokaryotic Cells and Viruses',
    'Analysis of Cell Components',
    'Cell Division - Mitosis',
    'Cell Division - Investigating Mitosis',
    'Cell Membrane Structure',
    'Exchange Across Cell Membranes - Diffusion',
    'Exchange Across Cell Membranes - Osmosis',
    'Exchange Across Cell Membranes - Active Transport',
    'The Immune System',
    'Immunity and Vaccines',
    'Antibodies in Medicine',
    'Interpreting Vaccine and Antibody Data',
    'HIV and Viruses',
    'Size and Surface Area',
    'Gas Exchange',
    'Gas Exchange in Humans',
    'The Effects of Lung Disease',
    'Interpreting Lung Disease Data',
    'Dissecting Gas Exchange Systems',


]

#for i in range(0,len(no_titles)):
#    print(no[i],no_titles[i])
yes = list(map(lambda x: x+5, yes))
no = list(map(lambda x: x - 1, no))
reader = PdfReader(r"../../CGP-AQA-Biology-A-Level-ariadanesh.com_.pdf")
aqabio_data = []
for i in range(0,len(yes)):
    aqabio_data.append((reader.pages[yes[i]].extract_text(),yes_titles[i],'Yes'))
for i in range(0,len(no)):
    aqabio_data.append((reader.pages[no[i]].extract_text(),no_titles[i],'No'))

#aqabio_data = [reader.pages[i].extract_text() for i in pages]
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------

# %% --------------------------------------------------------------------------
# cambridge business book
yes = [#add 7
    -1,
    2,3,15,29,38,51,61,70,76,98,124,125,137,159,170,187,200,212,213,231,252,273,297,318,330,331,339,366,376,385,403,416,417,434,443,464,476,486,495,505,518,534,535,543,556,568,587
]
yes_titles = ['Introduction',
              'Unit 1 Business and its environment',
              'Enterprise',
              'Business structure',
              'Size of business',
              'Business objectives',
              'Stakeholders in a business',
              'Business structure (A level)',
              'Size of a business (A level)',
              'External influences on business activity (A level)',
              'External economic influences on business behaviour (A level)',
              'Unit 2 People in organisations',
              'Management and leadership',
              'Motivation',
              'Human resource management',
              'Further human resource management (A Level)',
              'Organisation structure (A Level)',
              'Business communication (A Level)',
              'Unit 3 Marketing',
              'What is marketing?',
              'Market research',
              'The marketing mix - product and price',
              'The marketing mix - promotion and place',
              'Marketing planning (A Level)',
              'Globalisation and international marketing (A Level)',
              'Unit 4 Operations and project management',
              'The nature of operations',
              'Operations planning',
              'Inventory management',
              'Capacity utilisation (A Level)',
              'Lean production and quality management (A Level)',
              'Project management (A Level)',
              'Unit 5 Finance and accounting',
              'Business finance',
              'Costs',
              'Accounting fundamentals',
              'Forecasting and managing cash flows',
              'Costs (A levels)',
              'Budgets (A Level)',
              'Contents of published accounts (A Level)',
              'Analysis of published accounts (A Level)',
              'Investment appraisal (A Level)',
              'Unit 6 Strategic management',
              'What is strategic management? (A Level)',
              'Strategic analysis (A Level)',
              'Strategic choice (A Level)',
              'Strategic implementation (A Level)',
              'Preparing for your examinations',
              ]

no = [# minus 1
    4,4,7,7,8,8,9,10,11,23,37,57,59,72,82,90,111,132#check
    ,133,145,176,180,195,211,220,221,239,279,281,319,328,338,339,363,374,388,393,423,424,425,449,451,475,493,494,503,514,526,542,543,551,564,576,
]

no_titles = [
    'Business structure',
    'Unit 1 Business and its environment',
    'Business communication (A Level)',
    'External influences on business activity (A level)',
    'Introduction',
    'Understanding Strategy',
    'Unit 6 Strategic management',
    'Enterprise',
    'Business structure',
    'Size of business',
    'Business objectives',
    'Stakeholders in a business',
    'Business structure (A level)',
    'Size of a business (A level)',
    'External influences on business activity (A level)',
    'External economic influences on business behaviour (A level)',
    'Unit 2 People in organisations',
    'Management and leadership',
    'Motivation',
    'Human resource management',
    'Further human resource management (A Level)',
    'Organisation structure (A Level)',
    'Business communication (A Level)',
    'Unit 3 Marketing',
    'What is marketing?',
    'Market research',
    'The marketing mix - product and price',
    'The marketing mix - promotion and place',
    'Marketing planning (A Level)',
    'Globalisation and international marketing (A Level)',
    'Unit 4 Operations and project management',
    'The nature of operations',
    'Operations planning',
    'Inventory management',
    'Capacity utilisation (A Level)',
    'Lean production and quality management (A Level)',
    'Project management (A Level)',
    'Unit 5 Finance and accounting',
    'Business finance',
    'Costs',
    'Accounting fundamentals',
    'Forecasting and managing cash flows',
    'Costs (A levels)',
    'Budgets (A Level)',
    'Contents of published accounts (A Level)',
    'Analysis of published accounts (A Level)',
    'Investment appraisal (A Level)',
    'Unit 6 Strategic management',
    'What is strategic management? (A Level)',
    'Strategic analysis (A Level)',
    'Strategic choice (A Level)',
    'Strategic implementation (A Level)',
    'Preparing for your examinations',

]
#pages = list(map(lambda x: x - 1, pages))
#reader = PdfReader(r"../../cambridge business textbook.pdf")
#biscam_data = [reader.pages[i].extract_text() for i in pages]

for i in range(0,len(no_titles)):
    print(no[i],no_titles[i])

yes = list(map(lambda x: x+7, yes))
no = list(map(lambda x: x - 1, no))
reader = PdfReader(r"../../cambridge business textbook.pdf")
biscam_data = []
for i in range(0,len(yes)):
    biscam_data.append((reader.pages[yes[i]].extract_text(),yes_titles[i],'Yes'))
for i in range(0,len(no)):
    biscam_data.append((reader.pages[no[i]].extract_text(),no_titles[i],'No'))



# %%
# edexcel biology pearson
yes = [
    -4,
    -2,
    0, #these 3 diff papges
    2,4,8,11,14,16,22,26,28,31,37,41,44,50,52,54,56,59,63,65,67,70 #add 9
]
yes_titles = [
    'ABOUT THIS BOOK',
    'PRACTICAL SKILLS',
    'ASSESSMENT OVERVIEW',
    '1A CHEMISTRY FOR BIOLOGISTS',
    '1 THE CHEMISTRY OF LIFE',
    '2 CARBOHYDRATES 1: MONOSACCHARIDES AND DISACCHARIDES',
    '3 CARBOHYDRATES 2: POLYSACCHARIDES',
    '4 LIPIDS',
    '5 PROTEINS',
    'THINKING BIGGER: TREHALOSE',
    '1B MAMMALIAN TRANSPORT SYSTEMS',
    '1 THE PRINCIPLES OF CIRCULATION',
    '2 THE ROLES OF THE BLOOD',
    '3 CIRCULATION IN THE BLOOD VESSELS',
    '4 THE MAMMALIAN HEART',
    '5 ATHEROSCLEROSIS',
    '1C CARDIOVASCULAR HEALTH AND RISK',
    '1 RISK, CORRELATION AND CAUSE',
    '2 INVESTIGATING THE CAUSES OF CVDs',
    '3 RISK FACTORS FOR CARDIOVASCULAR DISEASE',
    '4 DIET AND CARDIOVASCULAR HEALTH',
    '5 DIETARY ANTIOXIDANTS AND CARDIOVASCULAR DISEASE',
    '6 USING THE EVIDENCE',
    '7 THE BENEFITS AND RISKS OF TREATMENT',
    'THINKING BIGGER: HEART FAILURE',
]

no = [
    2,2,6,6,7,7,8,9,9,10,10,11,11,
    12,13,
    14,18,21,24,26,32,36,40,41,50,53,54,60,62,64,66,71,73,75,79
]

no_titles = [
    'ABOUT THIS BOOK',
    '21 CHEMISTRY',
    'SPECIFICATION SUMMARY',
    'INTRODUCTION',
    'THINKING BIGGER: ENZYMES',
    '2 THE ROLES OF THE BLOOD',
    'ASSESSMENT OVERVIEW',
    '1 THE CELL WALL',
    '1A CHEMISTRY FOR BIOLOGISTS',
    'STATISTICS',
    '1 THE CHEMISTRY OF LIFE',
    'CARBOHYDRATES',
    'DATA AND METHODS',
    '1 THE CHEMISTRY OF LIFE',
    'IONIC AND COVALENT BONDING',
    '2 CARBOHYDRATES 1: MONOSACCHARIDES AND DISACCHARIDES',
    '3 CARBOHYDRATES 2: POLYSACCHARIDES',
    '4 LIPIDS',
    '5 PROTEINS',
    'THINKING BIGGER: TREHALOSE',
    '1B MAMMALIAN TRANSPORT SYSTEMS',
    '1 THE PRINCIPLES OF CIRCULATION',
    '2 THE ROLES OF THE BLOOD',
    '3 CIRCULATION IN THE BLOOD VESSELS',
    '4 THE MAMMALIAN HEART',
    '5 ATHEROSCLEROSIS',
    '1C CARDIOVASCULAR HEALTH AND RISK',
    '1 RISK, CORRELATION AND CAUSE',
    '2 INVESTIGATING THE CAUSES OF CVDs',
    '3 RISK FACTORS FOR CARDIOVASCULAR DISEASE',
    '4 DIET AND CARDIOVASCULAR HEALTH',
    '5 DIETARY ANTIOXIDANTS AND CARDIOVASCULAR DISEASE',
    '6 USING THE EVIDENCE',
    '7 THE BENEFITS AND RISKS OF TREATMENT',
    'THINKING BIGGER: HEART FAILURE',


]

yes = list(map(lambda x: x+9, yes))
no = list(map(lambda x: x - 1, no))
reader = PdfReader(r"../../biology textbook.pdf")
edbio_data = []
for i in range(0,len(yes)):
    edbio_data.append((reader.pages[yes[i]].extract_text(),yes_titles[i],'Yes'))
for i in range(0,len(no)):
    edbio_data.append((reader.pages[no[i]].extract_text(),no_titles[i],'No'))

# %% --------------------------------------------------------------------------
# hodder
no = [
    2,2,2,3,4,4,7 #minus 1
]
no_titles = [
    'Introduction',
    '1 - Cell division',
    'Cell biology',
    'Biological systems',
    'Cell membranes',
    'Topic 1 - Gas',
    'bacteria'

]
no = list(map(lambda x: x - 1, no))
reader = PdfReader(r"../../AQA-8461-HODDER-SAMPLE.pdf")
hod_data = []
for i in range(0,len(no)):
    hod_data.append((reader.pages[no[i]].extract_text(),no_titles[i],'No'))

# %% --------------------------------------------------------------------------

combi = aqabio_data+biscam_data+edbio_data+hod_data
final = pd.DataFrame(combi, columns=['page', 'chapter_title', 'label'])
final.to_json('page_identification_data.json')


# %%
from transformers import AutoTokenizer
model_id = "TheBloke/Mistral-7B-v0.1-GPTQ"

tokenizer = AutoTokenizer.from_pretrained(
    model_id, use_fast=False, add_eos_token=True
)
# %%
#hi = aqabio_data[0] +aqabio_data[1]+aqabio_data[2]+aqabio_data[3]
#hi = biotext_data[11]+biotext_data[12]+biotext_data[13]+biotext_data[14]+biotext_data[15]
lens = []
for i in final:
    lens.append(len(tokenizer(i)['input_ids']))


# %%
plz = pd.read_json('synthetic_data_newest.json')
for i in range(0,len(plz)):
    if len(tokenizer(plz['pages'][i])['input_ids']) >= len(tokenizer(plz['sums'][i])['input_ids'])/0.6:
        print(i)


# %% --------------------------------------------------------------------------
plz['sums'][141]
# -----------------------------------------------------------------------------

# %%
#for first page test ...
yes = [
    1,10,25,44,55,73,90,102,116,128,145,173,188,210,221,247,274,296,310    
]
yes_titles=[
    '1 Quantities and units',
    '2 Practical skills',
    '3 Rectilinear motion',
    '4 Momentum',
    '5 Forces',
    '6 Work, energy and power',
    '7 Charge and current',
    '8 Potential difference, electromotive force and power',
    '9 Current-potential difference relationships',
    '10 Resistance and resistivity',
    '11 Internal resistance, series and parallel circuits, and the potential divider',
    '12 Fluids',
    '13 Solid materials',
    '14 Nature of waves',
    '15 Transmission and reflection of waves',
    '16 Superposition of waves',
    '17 Particle nature of light',
    '18 Maths in physics',
    '19 Preparing for the exams'
]

no = [
    1,2,2,4,4,6,6,7,7,8,8,9,9,10,11,12,20,35,54
]

no_titles = [
    '1 Fluids',
    '1 Quantities and units',
    'Practical skills',
    'Quantities and units',
    '10 Maths in physics',
    'Quantities',
    'Practical activity',
    'Exam preparation',
    '4 Momentum',
    '12 Fluids',
    '19 Preparing for the exams',
    'Exam preparation',
    '2 Practical skills',
    'Quantities and units',
    '13 Solid materials',
    '1 Quantities and units',
    '18 Maths in physics',
    '14 Nature of waves',
    '11 Internal resistance, series and parallel circuits, and the potential divider'


]

for i in range(0,len(yes_titles)):
    print(yes[i],yes_titles[i])

yes = list(map(lambda x: x+9, yes))
no = list(map(lambda x: x - 1, no))
reader = PdfReader(r"../../edexcel_a_level_physics_student_book_1.pdf")
test_data = []
for i in range(0,len(yes)):
    test_data.append((reader.pages[yes[i]].extract_text(),yes_titles[i],'Yes'))
for i in range(0,len(no)):
    test_data.append((reader.pages[no[i]].extract_text(),no_titles[i],'No'))

test_data = pd.DataFrame(test_data, columns=['page', 'chapter_title', 'label'])
test_data.to_json('firstpage_testdata.json')