# %% --------------------------------------------------------------------------
from PyPDF2 import PdfReader
import pandas as pd

# -----------------------------------------------------------------------------

# %% --------------------------------------------------------------------------
# aqa biology
[]
pages = [
    8,
    9,
    10,#
    12,13,#
    14,
    15,#
    16,
    17,#
    18,
    19,#
    20,21,#
    22,23,##
    24,25,##
    30,31,32,33,#
    36,37,##
    40,41,##
    42,43,##
    44,45,##
    46,47,##
    50,51,##
    52,53,##
    54,55,##
    58,59,##
    60,61,##
    62,
    64,65,##
    70,71,##
    72,73,##
    74,75,##
    76,77,##
    78,79,80,#
    81,82,83,#
    84,85,##
    86,87,##
    88,89,##
    90,91,##
    92,93,##
    94,95,96,#
    98,99,##
    100,101,##
    102,103,##
    104,105,##
    108,109,##
    110,111,##
    112,113,114,115,#
    120,121,122,123,#
    142,143,144,#
    145,146,147,#
    162,163,164,
    165,166,167,#
    192,193,194,195,#
    207,208,209,#
    214,215,#
    216,217,218,#
    219,220,



]
pages = list(map(lambda x: x - 1, pages))
reader = PdfReader(r"../../CGP-AQA-Biology-A-Level-ariadanesh.com_.pdf")
aqabio_data = [reader.pages[i].extract_text() for i in pages]
# -----------------------------------------------------------------------------

# %% --------------------------------------------------------------------------
#aqabio_data_new = []
aqabio_data_new.append(aqabio_data[21])
# -----------------------------------------------------------------------------

# %% --------------------------------------------------------------------------
aqabio_data_new.append(aqabio_data[68]+aqabio_data[69])
# -----------------------------------------------------------------------------

# %% --------------------------------------------------------------------------
aqabio_data_new.append(aqabio_data[112]+aqabio_data[113]+aqabio_data[114])
# -----------------------------------------------------------------------------

# %% --------------------------------------------------------------------------
aqabio_data_new.append(aqabio_data[103]+aqabio_data[104]+aqabio_data[105]+aqabio_data[106])
# -----------------------------------------------------------------------------

# %% --------------------------------------------------------------------------
aqabio_data_df = pd.DataFrame({'pages':aqabio_data_new})
aqabio_data_df.to_json('aqabio_data_mod.json')
# -----------------------------------------------------------------------------

# %% --------------------------------------------------------------------------
# cambridge business book
pages = [
    11,12,
    14,15,16, #14,15
    17,18,#17
    24,25,26,
    37,38,39,
    46,47,
    53,54,55,#53
    59,60,
    60,61,62,
    246,247,248,
    261,
    268,269,
    272,273,274,
    282,283,
    294,
    310,
    314,
    316,317,
    326,327,328


]
pages = list(map(lambda x: x - 1, pages))
reader = PdfReader(r"../../cambridge business textbook.pdf")
biscam_data = [reader.pages[i].extract_text() for i in pages]


# %% --------------------------------------------------------------------------

#biscam_data_new = []
biscam_data_new.append(biscam_data[36])
# -----------------------------------------------------------------------------

# %% --------------------------------------------------------------------------
biscam_data_new.append(biscam_data[37]+biscam_data[38])
# -----------------------------------------------------------------------------

# %% --------------------------------------------------------------------------
biscam_data_new.append(biscam_data[39]+biscam_data[40]+biscam_data[41])
# -----------------------------------------------------------------------------

# %% --------------------------------------------------------------------------
biscam_data_new.append(biscam_data[103]+biscam_data[104]+biscam_data[105]+biscam_data[106])
# -----------------------------------------------------------------------------

# %% --------------------------------------------------------------------------
biscam_data_df = pd.DataFrame({'pages':biscam_data_new})
biscam_data_df.to_json('biscam_data_mod.json')


# %%
# edexcel biology pearson
pages = [
    14,15,16,17,
    18,19,
    21,22,23,
    24,25,
    26,27,28,29,30,
    38,39,40,
    41,42,43,44,
    45,
    47,48,49,
    51,52,53,
    54,55,56,57,
    62,63,
    64,65,
    66,67,68,
    69,70,
    75,
    77,78,79,
]
pages = list(map(lambda x: x - 1, pages))
reader = PdfReader(r"../../biology textbook.pdf")
biotext_data = [reader.pages[i].extract_text() for i in pages]

# %% --------------------------------------------------------------------------

#biotext_data_new = []
biotext_data_new.append(biotext_data[43])
# -----------------------------------------------------------------------------

# %% --------------------------------------------------------------------------
biotext_data_new.append(biotext_data[12]+biotext_data[13])
# -----------------------------------------------------------------------------

# %% --------------------------------------------------------------------------
biotext_data_new.append(biotext_data[44]+biotext_data[45]+biotext_data[46])
# -----------------------------------------------------------------------------

# %% --------------------------------------------------------------------------
biotext_data_new.append(biotext_data[30]+biotext_data[31]+biotext_data[32]+biotext_data[33])
# -----------------------------------------------------------------------------

# %% --------------------------------------------------------------------------
biotext_data_df = pd.DataFrame({'pages':biotext_data_new})
biotext_data_df.to_json('biotext_data_mod.json')


# %% --------------------------------------------------------------------------
# edexcel biology pearson
pages = [
    8,9,
    10,11,12,
    13,14,15,16,#13,14
    16,17,18,
]
pages = list(map(lambda x: x - 1, pages))
reader = PdfReader(r"../../AQA-8461-HODDER-SAMPLE.pdf")
hod_data = [reader.pages[i].extract_text() for i in pages]


# %% --------------------------------------------------------------------------

#hod_data_new = []
hod_data_new.append(hod_data[43])
# -----------------------------------------------------------------------------

# %% --------------------------------------------------------------------------
hod_data_new.append(hod_data[2]+hod_data[3])
# ----------------------------------------------------------------------------

# %% --------------------------------------------------------------------------
hod_data_new.append(hod_data[9]+hod_data[10]+hod_data[11])
# -----------------------------------------------------------------------------

# %% --------------------------------------------------------------------------
hod_data_new.append(hod_data[5]+hod_data[6]+hod_data[7]+hod_data[8])
# -----------------------------------------------------------------------------

# %% --------------------------------------------------------------------------
hod_data_df = pd.DataFrame({'pages':hod_data_new})
hod_data_df.to_json('hod_data_mod.json')
# -----------------------------------------------------------------------------

# %% combne all .....
data = pd.concat([pd.read_json('aqabio_data_mod.json'),pd.read_json('biotext_data_mod.json'),pd.read_json('biscam_data_mod.json'),pd.read_json('hod_data_mod.json')])
data.reset_index(inplace=True)
data.drop(columns='index',inplace=True)
data

# %%

data.to_json(r'data/revamped_data.json')

# %%
from transformers import AutoTokenizer
model_id = "TheBloke/Mistral-7B-v0.1-GPTQ"

tokenizer = AutoTokenizer.from_pretrained(
    model_id, use_fast=False, add_eos_token=True
)
# %%
#hi = aqabio_data[0] +aqabio_data[1]+aqabio_data[2]+aqabio_data[3]
#hi = biotext_data[11]+biotext_data[12]+biotext_data[13]+biotext_data[14]+biotext_data[15]
hi = '1BSPECIFICATION \nREFERENCE1.73  CIR CULATION IN THE  \nBLOOD VESSELS\nLEARNING OBJECTIVES\n ◼Under stand how the structures of blood vessels (arteries, veins and capillaries) relate to their functions.\nTHE BLOOD VESSELS\nThe blood vessels that make up the circulatory system can be thought of  as the biological equivalent \nof  a road transport system. The arteries and veins are like the large roads carrying heavy traffic while the narrow town streets and tracks are represented by the vast area of  branching and spreading capillaries called the capillary network. In the capillary network, substances carried by the blood are exchanged with cells in the same way that products are transported from factories, oil refineries or farms and distributed into shops and homes. The structures of  the different types of  blood vessel closely reflect their functions in your body.\nARTERIES\nArteries carry blood away from your heart towards the cells of  your body. The structure of  an artery is shown in fig A. Almost all arteries carry oxygenated blood. The exceptions are:\n •the pulmonar\ny artery – carrying deoxygenated blood from the heart to the lungs\n •the umbilical ar\ntery – during pregnancy, this carries deoxygenated blood from the fetus to  \nthe placenta.\nThe arteries leaving the heart branch off  in every direction, and the diameter of  the lumen, the central space inside the blood vessel, gets smaller the further away it is from the heart. The very smallest branches of  the arterial system, furthest from the heart, are the arterioles. \nThe middle layers of\nthe artery wall containelastic ﬁbres and smoothmuscle; arteries nearestthe heart have more elasticﬁbres, those further fromthe heart have a greaterproportion of muscle tissue.\nelastic ﬁbres andsmooth musclelumentough outerlayer\nendotheliumexternal layerof tough tissue\nThe endothelium forms asmooth lining which allows theeasiest possible ﬂow of blood.Lumen is small whenartery unstretched byﬂow of blood from heart.\n▲ fig A  T he structure of an artery means it is adapted to cope with the surging of the blood as the heart pumps.\nBlood is pumped out from the heart in a regular rhythm, about 70 times a minute. Each heartbeat \nsends a high-pressure flow of  blood into the arteries. The major arteries close to the heart must withstand these pressure surges. Their walls contain a lot of  elastic fibres, so they can stretch to accommodate the greater volume of  blood without being damaged (see fig B). Between surges, the elastic fibres return to their original length, squeezing the blood to move it along in a continuous flow. The pulse you can feel in an artery is the effect of  the surge each time the heart beats. The blood pressure in all arteries is relatively high, but it falls in arteries further away from the heart. These are known as the peripheral arteries. LEARNING TIP\nRemember that all arteries carry \nblood away from the heart, so they have thick walls and lots of collagen to withstand the high pressure.\nEXAM HINT\nYou will study the structure and the function of the types of blood vessel separately. However, you should remember that the vessels do not exist separately – they are all interlinked within the whole circulatory system.\nUncorrected proof, all content subject to change at publisher discretion. Not for resale, circulation or distribution in whole or in part. ©Pearson 2018MAMMALIAN TRANSPORT SYSTEMS 38 1B.3 CIRCULATION IN THE BLOOD VESSELS\nEXAM HINT\nThe role of the elastic fibres in artery walls is to return to their original \nlength to help maintain the pressure. This is called recoil. The elastic recoil does not help to increase pressure, it simply helps to maintain the pressure – so do not suggest that the recoil helps pump blood along.\nIn the peripheral arteries, the muscle fibres in the vessel walls \ncontract or relax to change the size of  the lumen, controlling the blood flow. The smaller the lumen, the harder it is for blood to flow through the vessel. This controls the amount of  blood that flows into an organ, so regulating its activity. Y ou will find out more about this important response in Book 2 Topic 7.\n2.5 cmaorta\nelastic ﬁbres\nsmooth muscle\ncollagen ﬁbres\nelastic ﬁbres\nsmooth muscle\ncollagen ﬁbres\nelastic ﬁbres\nsmooth muscle\ncollagen ﬁbresmedium-sized artery\narteriole0.4 cm\n30 /uni03BCm\n▲ fig B  T he relative proportions of different tissues in different arteries. \nCollagen gives general strength and flexibility to both arteries and veins.\nLEARNING TIP\nThe role of the muscles in the wall of the arterioles is to reduce the size \nof the lumen to increase resistance – this can reduce blood flow to areas that do not need so much blood and will cause the oxygenated blood to flow to other tissues. Remember to link this to things you may learn later such as how blood flow to the skin changes when you are too hot or too cold.\nCAPILLARIES\nArterioles lead into networks of  capillaries. These are very small \nvessels that spread throughout the tissues of  the body. The capillary network links the arterioles and the venules. Capillaries branch between cells – no cell is far from a capillary, so substances can diffuse between cells and the blood quickly. Also, because the diameter of  each individual capillary is small, the blood travels relatively slowly through them, giving more opportunity for diffusion to occur (see fig C). The smallest capillary is no wider than a single red blood cell.\nCapillaries have a very simple structure which is well adapted \nto their function. Their walls are very thin and contain no elastic fibres, smooth muscle or collagen. This helps them fit between individual cells and allows rapid diffusion of  substances between the blood and the cells. The walls consist of  just one very thin cell. Oxygen and other molecules, such as digested food molecules and hormones, quickly diffuse out of  the blood in the capillaries into the nearby body cells, and carbon dioxide and other waste molecules diffuse into the capillaries. Blood entering the capillary network from the arteries is oxygenated. When it leaves, it carries less oxygen and more carbon dioxide.\nwaste material \ne.g. carbon dioxideoxygen andfood molecules\ncapillary wall(epithelial cells)\nsingle red blood cells\n▲ fig C  T he very thin walls of capillaries allow rapid diffusion of oxygen, \ncarbon dioxide and digested food molecules. The lumen is just wide enough for red blood cells to pass through.\nVEINS\nVeins carry blood back towards the heart. Most veins carry \ndeoxygenated blood. The exceptions are:\n •the pulmonar\ny vein – carrying oxygen-rich blood from the \nlungs back to the heart for circulation around the body\n •the umbilical v\nein – during pregnancy, it carries oxygenated \nblood from the placenta into the fetus.\nTiny venules lead from the capillary network, combining into larger and larger vessels going back to the heart (see fig D).\nLEARNING TIP\nRemember that all veins carry blood back to the heart so they have \nlow pressure and do not need a thick wall.\nUncorrected proof, all content subject to change at publisher discretion. Not for resale, circulation or distribution in whole or in part. ©Pearson 201839 1B.3 CIRCULATION IN THE BLOOD VESSELS MAMMALIAN TRANSPORT SYSTEMS\nsmooth inner surface\nouter tough layer consisting\nmainly of collagen ﬁbresrelatively thin layer of smooth muscle with few elastic ﬁbres\nrelatively large lumen\nvein\nartery\n▲ fig D  T he arrangement of tissues in a vein reflects the pressure of blood  \nin the vessel.\nEventually only two veins (sometimes called the great veins) carry \nthe blood from the body tissues back to the heart – the inferior vena cava from the lower parts of  the body and the superior vena cava from the upper parts of  the body.\nVeins can hold a large volume of  blood – in fact more than half  of  \nthe body’s blood volume is in the veins at any one time. They act as a blood reservoir. The blood pressure in the veins is relatively low – the pressure surges from the heart are eliminated before the blood reaches the capillary system. This blood at low pressure must be returned to the heart and lungs to be oxygenated again and recirculated.\nThe blood is not pumped back to the heart, it returns to the heart \nby means of  muscle pressure and one-way valves.\n •Man\ny of  the larger veins are situated between the large muscle \nblocks of  the body, particularly in the arms and legs. When  the muscles contract during physical activity they squeeze these veins. The valves (see below) keep the blood travelling  in one direction and this squeezing helps to return the blood  to the heart. \n •T\nhere are one-way valves at frequent intervals throughout \nthe venous system. These are called semilunar valves because of  their half-moon shape. They develop from infoldings of  the inner wall of  the vein. Blood can pass through towards the heart, but if  it starts to flow backwards the valves close, preventing any backflow (see fig E).semilunar\nvalve open muscle\ncontracte d\nvein\nveinmusclerelaxedsemilunar valve shutBlood moving in the direction of the heart forces the valve open, allowing the blood to ﬂow through.\nA backﬂow of blood will close the valve, ensuring \nthat blood cannot ﬂow away from the heart.\n▲ fig E  V alves in the veins make sure blood only flows in one direction – \ntowards the heart. The contraction of large muscles encourages blood  flow through the veins.\nThe main types of  blood vessel – the arteries, veins and capillaries \n– have very different characteristics. These affect the way the blood flows through the body, and what the vessels do in the body. Some of  these differences are summarised in fig F.\nhigh\nlowtotal area\n(cm2)total area(cm\n2)blood pressure(kPa)\nvelocity(cm s\n21)\nlarge\narteriessmall\narteriesarterioles capillaries venules veins\n▲ fig F  Gr aph to show the surface area of each major type of blood vessel \nin your body, along with the velocity and pressure of the blood travelling  in them.\nUncorrected proof, all content subject to change at publisher discretion. Not for resale, circulation or distribution in whole or in part. ©Pearson 2018'

bye = tokenizer(hi)
len(bye['input_ids'])
# %%
