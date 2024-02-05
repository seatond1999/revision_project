
# %% --------------------------------------------------------------------------
import pandas as pd
import fitz 


# %% --------------------------------------------------------------------------

def read_pdf(file_path,page_number):
    with fitz.open(file_path) as pdf_document:
        page = pdf_document[page_number]
        text = page.get_text()

    return text

# Replace 'your_pdf_file.pdf' with the path to your PDF file

# -----------------------------------------------------------------------------

# %%
#9781398332447-Pearson-Edexcel-A-Level-Geography-2-Human-Geography-Workbook
yes = [
    1
]

no = [
    0,2
]

synthetic = [
    'Contents\n3 Internationalization ...................................6\n• What initiates the process of internationalization and why \nhas it gained momentum in recent years? \n• What effects does internationalization have on \ncountries, diverse social groups, and \ncultures, as well as the physical environment?\n• What are the implications of internationalization for global \ndevelopment and the physical \nenvironment, and how should various stakeholders \naddress its challenges?\nExam-style questions ................................18\n4A Renewing localities ....................20\n• How and why do localities differ?\n• Why might rejuvenation be necessary?\n• How is the management of rejuvenation handled?\n• To what extent is rejuvenation successful?\nExam-style questions ................................37\n4B Varied localities ...............................41\n• How do demographic structures differ?\n• How do diverse communities perceive various residential \nareas?\n• Why do demographic and cultural \nfrictions arise in varied localities?\n• How effectively are cultural and \ndemographic issues being managed?\nExam-style questions ................................57\n7 Dominant Powers .................................61\n• What characterizes dominant powers and how have they \ntransformed over time?\n• What are the influences of dominant powers on the \nglobal economy, political systems, and the \nphysical environment?\n• Which areas of influence are contested by \ndominant powers and what are the repercussions of \nthis?\nExam-style questions ................................71\n8A Well-being, human rights and \ninterference....................................73\n• What is human well-being and why does it differ from place to place?\n• Why do human rights vary from place to \nplace?\n• How are human rights used as justifications for \npolitical and military intervention?\n• What are the outcomes of geopolitical \ninterventions in terms of human well-being \nand human rights?\nExam-style questions ................................85\n8B Movement, identity and \nautonomy ....................................88\n• What are the consequences of internationalization on \ntransnational migration?\n• How are nation states delineated and how have \nthey transformed in a globalized world?\n• What are the influences of global organizations \non addressing global issues and conflicts?\n• What are the challenges to national autonomy in \na more globalized world?\nExam-style questions ..............................101\n',
    'Contents\n3 Global Integration ...................................5\n• What triggers the process of global integration and why \nhas it gained momentum in recent times? \n• What effects does global integration have on \ncountries, diverse social groups, and \ncultures, as well as the physical environment?\n• What are the implications of global integration for global \ndevelopment and the physical \nenvironment, and how should various actors \nrespond to its challenges?\nExam-style questions ................................17\n4A Revitalizing localities ....................19\n• How and why do localities differ?\n• Why might revitalization be necessary?\n• How is revitalization managed?\n• To what extent is revitalization successful?\nExam-style questions ................................36\n4B Assorted localities ...............................40\n• How do population structures differ?\n• How do diverse communities perceive varied residential \nareas?\n• Why do demographic and cultural \ntensions arise in assorted localities?\n• How effectively are cultural and \ndemographic issues being managed?\nExam-style questions ................................56\n7 Global Powers .................................60\n• What characterizes global powers and how have they \ntransformed over time?\n• What are the impacts of global powers on the \nglobal economy, political systems, and the \nphysical environment?\n• Which areas of influence are contested by \nglobal powers and what are the consequences of \nthis?\nExam-style questions ................................70\n8A Health, human rights and \nintervention....................................72\n• What is human development and why do \nlevels vary from place to place?\n• Why do human rights vary from place to \nplace?\n• How are human rights used as justifications for \npolitical and military intervention?\n• What are the outcomes of geopolitical \ninterventions in terms of human development \nand human rights?\nExam-style questions ................................84\n8B Migration, identity and \nsovereignty ....................................87\n• What are the impacts of global integration on \ninternational migration?\n• How are nation states defined and how have \nthey evolved in a globalizing world?\n• What are the impacts of global organizations \non managing global issues and conflicts?\n• What are the threats to national sovereignty in \na more globalized world?\nExam-style questions ..............................100\n'
]

pdf_path = r'../../../../new books/9781398332447-Pearson-Edexcel-A-Level-Geography-2-Human-Geography-Workbook.pdf'
page = 1
read_pdf(pdf_path,page)

# %% --------------------------------------------------------------------------
#9781446930885-gce2015-a-bioa-spec 
yes = [
    10,11
]

no = [
    2,4,6,7,12,9,14
]

synthetic = [
    " \n \n \nContents \nProgram Overview \n1 \nUnderstanding and Competence \n6 \nScientific Practical Validation \n27 \nGrading and Calibration \n31 \nEthical Standards \n32 \nAppraisal \n34 \nSummary of Evaluations \n34 \nObjectives and Allocations of Evaluations \n36 \nAnalysis of Evaluation Objectives \n36 \nEntry and Evaluation Details \n37 \nStudent Registration Procedures \n37 \nPromotion Codes and Result Metrics \n37 \nAccess Arrangements, Rational Adjustments, and Special \nConsiderations \n38 \nEquality Act 2010 and Pearson's Equality Guidelines \n39 \nIntegrated Evaluation \n39 \nRecognition and Result Presentation \n40 \nLanguage Used in Assessment \n40 \nAdditional Information \n42 \nStudent Recruitment Strategies \n42 \nPrevious Learning and Additional Requirements \n42 \nAdvancement Paths \n42 \nInterconnection between Preliminary GCE and \nHigher-level GCE \n42 \nAdvancement from Preliminary GCE to Higher-level \nGCE \n43 \nInterconnection between General Certificate of Secondary Education (GCSE) and Higher-level GCE \n43 \nAdvancement from GCSE to Higher-level GCE \n43 \nAppendix 1: Adaptable Skills \n46 \nAppendix 2: Level 3 Extended Project Qualification \n48 \nAppendix 3: Coding Protocols \n52 \nAppendix 4: Verification of Practical Competence \nSheet \n54 \nAppendix 5: Methodology in Scientific Work \n56 \nAppendix 5a: Practical Abilities Recognized for Indirect \nAssessment and Developed through Teaching and \nLearning \n58 \n",
    "\n \n \nContents \nQualification Overview \n1 \nInsight, Proficiency, and Knowledge \n5 \nScientific Practical Approval \n28 \nEvaluation and Grading \n32 \nEthical Behavior \n33 \nAppraisal Overview \n35 \nSummary of Evaluations \n35 \nObjectives and Weightings of Evaluations \n37 \nAnalysis of Evaluation Objectives \n37 \nEntry and Evaluation Details \n38 \nStudent Entry Procedures \n38 \nPromotion Codes and Result Metrics \n38 \nAccess Arrangements, Rational Adjustments, and Special \nConsiderations \n39 \nEquality Act 2010 and Pearson Equality Policies \n40 \nComprehensive Evaluation \n40 \nRecognition and Outcome Presentation \n41 \nLanguage Used in Appraisal \n41 \nAdditional Information \n43 \nStudent Recruitment Approaches \n43 \nPrevious Learning and Additional Prerequisites \n43 \nAdvancement Strategies \n43 \nInterrelation between Advanced Subsidiary GCE and \nAdvanced GCE \n43 \nAdvancement from Advanced Subsidiary GCE to Advanced \nGCE \n44 \nInterrelation between GCSE and Advanced GCE \n44 \nAdvancement from GCSE to Advanced GCE \n44 \nAppendix 1: Transferable Proficiencies \n47 \nAppendix 2: Level 3 Extended Project Qualification \n49 \nAppendix 3: Coding Protocols \n53 \nAppendix 4: Verification of Practical Competence \nSheet \n55 \nAppendix 5: Scientific Methodology \n57 \nAppendix 5a: Practical Skills Recognized for Indirect \nAssessment and Developed through Teaching and \nLearning \n59 \n",
    "\n \nAppendix 5b: Practical Proficiencies for Direct \nAssessment and Cultivation through Teaching and \nLearning \n60 \nAppendix 5c: Utilization of Apparatus and Techniques \n62 \nAppendix 5d: Correspondence between Appendix 5c and \nCore Practicals (Biology) \n64 \nAppendix 6: Mathematical Abilities and \nIllustrations \n68 \nAppendix 7: Terminology in Examination \nDocuments \n74 \nAppendix 8: Presentation by Topics \n76 \nAppendix 9: Assistance from the University of York \n96 \n \n",
    "\n \nAppendix 5b: Practical Abilities for Direct \nEvaluation and Enhancement through Teaching and \nLearning \n61 \nAppendix 5c: Application of Instruments and Methods \n63 \nAppendix 5d: Correspondence between Appendix 5c and \nEssential Practicals (Biology) \n65 \nAppendix 6: Mathematical Competencies and \nIllustrations \n69 \nAppendix 7: Directives in Examination \nDocuments \n75 \nAppendix 8: Display by Themes \n77 \nAppendix 9: Assistance from the University of York \n97 \n \n"
    ]

file = '9781446930885-gce2015-a-bioa-spec.pdf'
pdf_path = rf'../../../../new books/{file}'
page = 11
read_pdf(pdf_path,page)
# -----------------------------------------------------------------------------

# %%
#A_level_History_interpretations_guidance_abridged
yes = [
    1
]

no = [
    2,5,0
]

synthetic = [
    "\n \n \n2 \n \n \n \nContents \nOverview \n3 \nThe Nature of Historical Perspectives \n5 \nThe Mirage of Ultimate History \n5 \nHistory as a Collective Endeavor \n5 \nChallenges in Comprehending History \n7 \nStudies on the Cognitive Development of Younger Children in Historical Interpretation \n7 \nStudies on the Thought Process of 16-19 Year-Old Students Regarding Interpretations \n8 \nApproaches \n9 \nInterpretations, Representations, and Constructs \n10 \nAnalyzing Varied Historians' Perspectives \n11 \nAssessing Interpretations Using Appropriate Criteria \n12 \nExercises \n15 \nAuthor's Profile \n24 \nAppreciation \n24 \n \n",
    "\n \n \n2 \n \n \n \nContents \nOverview \n3 \nThe Concept of Historical Interpretations \n5 \nThe Myth of Definitive History \n5 \nHistory as a Collective Endeavor \n5 \nChallenges in Comprehending History \n7 \nResearch on Cognitive Development in Younger Children Regarding Historical Interpretation \n7 \nResearch on 16-19 Year-Old Students' Perspectives on Interpretations \n8 \nApproaches and Techniques \n9 \nInterpretations, Representations, and Constructions \n10 \nExplaining Discrepancies in Historians' Perspectives \n11 \nAssessing Interpretations Using Appropriate and Relevant Criteria \n12 \nPractical Activities \n15 \nAuthor's Biography \n25 \nAcknowledgements \n25 \n \n"
    ]

file = 'A_level_History_interpretations_guidance_abridged.pdf'
pdf_path = rf'../../../../new books/{file}'
page = 1
read_pdf(pdf_path,page)
# %%
