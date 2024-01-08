
# %% --------------------------------------------------------------------------
import pandas as pd
from openai import OpenAI
import os
import time
import openai
import requests

# ----------------------------------------------------------------------------
df = pd.read_json(r'final_pages.json')
df.reset_index(inplace=True)
df.drop(columns='index',inplace=True)
df
# %% --------------------------------------------------------------------------
def gpt_enrich_data(data_inp, start_row=0, end_row=None):
    #end_row = len(data_inp) if end_row is None else end_row
    data = data_inp
    prompt1 = f"""You will sort text into subsections and add informative titles for each subsection, without removing any information from the original text.
Each subsection must be in paragraph form and no information should be missing from the original text.
Make subsections as large as possible.
Mark each title you create by adding the symbols "@@@" before each title. For example, if i gave you this text: "17 1A.5 PROTEINS CHEMISTRY FOR BIOLOGISTS\nHYDROGEN BONDS\nY ou were introduced to hydrogen bonds in Section 1A.1. These same bonds are essential in protein \nstructures. In amino acids, tiny negative charges are present on the oxygen of  the carboxyl groups and tiny positive charges are present on the hydrogen atoms of  the amino groups. When these charged groups are close to each other, the opposite charges attract, forming a hydrogen bond. Hydrogen bonds are weak but, potentially, they can be made between any two amino acids in the correct position, so there are many of  them holding the protein together very firmly. They are very important in the folding and coiling of  polypeptide chains (see fig C). Hydrogen bonds break easily and reform if  pH or temperature conditions change.\nDISULFIDE BONDS\nDisulfide bonds form when two cysteine molecules are close together in the structure of  a polypeptide (see fig C). An oxidation reaction occurs between the two sulfur-containing groups, resulting in a strong covalent bond known as a disulfide bond. These disulfide bonds are much stronger than hydrogen bonds but they happen much less often. They are important for holding the folded polypeptide chains in place.\nhydrogen bondhydrogen bond\ndisulﬁde bonddisulﬁde bond\nα-helixα-helixα-helixα-helixβ-pleated sheetβ-pleated sheetβ-pleated sheetβ-pleated sheet\n▲ fig C  Hydr ogen bonds and disulfide bonds maintain the shape of protein molecules and this determines  \ntheir function." I would expect you to give me an output like this: "@@@HYDROGEN BONDS
You were introduced to hydrogen bonds in Section 1A.1. These same bonds are essential in protein structures. In amino acids, tiny negative charges are present on the oxygen of the carboxyl groups and tiny positive charges are present on the hydrogen atoms of the amino groups. When these charged groups are close to each other, the opposite charges attract, forming a hydrogen bond. Hydrogen bonds are weak but, potentially, they can be made between any two amino acids in the correct position, so there are many of them holding the protein together very firmly. They are very important in the folding and coiling of polypeptide chains (see fig C). Hydrogen bonds break easily and reform if pH or temperature conditions change.

@@@DISULFIDE BONDS
Disulfide bonds form when two cysteine molecules are close together in the structure of a polypeptide (see fig C). An oxidation reaction occurs between the two sulfur-containing groups, resulting in a strong covalent bond known as a disulfide bond. These disulfide bonds are much stronger than hydrogen bonds but they happen much less often. They are important for holding the folded polypeptide chains in place.
hydrogen bondhydrogen bond
disulﬁde bonddisulﬁde bond
α-helixα-helixα-helixα-helixβ-pleated sheetβ-pleated sheetβ-pleated sheetβ-pleated sheet
▲ fig C Hydrogen bonds and disulfide bonds maintain the shape of protein molecules and this determines their function." Do you understand the task?"""
    response1 = 'Yes, I understand the task. Please provide the text you would like me to sort into subsections and add informative titles for each subsection, without removing information.'

    data_to_enrich = data.iloc[
        start_row:end_row
    ]  # i dont pass all due to open AI limits

    os.environ["OPENAI_API_KEY"] = "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"
    key = "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"
    openai.api_key = key
    client = OpenAI()
    subs = []
    try:
        for i in range(0, len(data_to_enrich)):
            print(i)

            prompt2 = f"This is the text: {data_to_enrich.iloc[i,0]}"

            url = "https://api.openai.com/v1/chat/completions"
            headers = {
                "Content-Type": "application/json",
                "Authorization": "Bearer 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"
            }

            data = {
                "model": "gpt-4-1106-preview",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant who will follow the instructions of the user"
                    },
                    {
                        "role": "user",
                        "content": f"{prompt1}"
                    },
                    {
                        "role": "assistant",
                        "content": f"{response1}"
                    },
                    {
                        "role": "user",
                        "content": f"{prompt2}"
                    }
                ],
                'temperature':1
            }
            response = requests.post(url, headers=headers, json=data)

            # Print the response content
            res = (response.json())
            subs.append(res['choices'][0]['message']['content'])

            print(i)
            #time.sleep(62) if (i + 1) % 3 == 0 else None
    except:
        None

    return subs
    
# -----------------------------------------------------------------------------
# %%

data = gpt_enrich_data(df)

#1st point = ["@@@TOPIC IA — BIOLOGICAL MOLECULES: CARBOHYDRATES AND COMMON BIOCHEMISTRY\nEven though there is, and has been, a huge variety of different organisms on Earth, they all share some biochemistry— for example, they all contain a few carbon-based compounds that interact in similar ways.\n\n@@@MOST CARBOHYDRATES ARE POLYMERS\n1) Most carbohydrates (as well as proteins and nucleic acids) are polymers.\n2) Polymers are large, complex molecules composed of long chains of monomers joined together.\n3) Monomers are small, basic molecular units.\n4) Examples of monomers include monosaccharides, amino acids and nucleotides.\nmonomer e.g. monosaccharide, amino acid polymer e.g. carbohydrate, protein\n\n@@@CARBOHYDRATES ARE MADE FROM MONOSACCHARIDES\n1) All carbohydrates contain the elements C, H and O.\n2) The monomers that they're made from are monosaccharides, e.g. glucose, fructose and galactose.\n1) Glucose is a hexose sugar — a monosaccharide with six carbon atoms in each molecule.\n2) There are two types of glucose, alpha (α) and beta (β)— they're isomers (molecules with the same molecular formula as each other, but with the atoms connected in a different way).\n3) You need to know the structures of both types of glucose for your exam— it's pretty easy because there's only one difference between the two:\nα-glucose molecule β-glucose molecule\nThe two types of glucose have these groups reversed\n\n@@@CONDENSATION REACTIONS JOIN MONOSACCHARIDES TOGETHER\n1) A condensation reaction is when two molecules join together with the formation of a new chemical bond, and a water molecule is released when the bond is formed.\n2) Monosaccharides are joined together by condensation reactions.\n3) A glycosidic bond forms between the two monosaccharides as a molecule of water is released.\n4) A disaccharide is formed when two monosaccharides join together.\nExample\nTwo α-glucose molecules are joined together by a glycosidic bond to form maltose.\n5) Sucrose is a disaccharide formed from a condensation reaction between a glucose molecule and a fructose molecule.\n6) Lactose is another disaccharide formed from a glucose molecule and a galactose molecule.\nIf you're asked to show a condensation reaction, don't forget to put the water molecule in as a product."]
# %%
final = pd.DataFrame({'pages':df['pages'],'sums':data})
final.to_json(r"synthetic_data_newest.json")
#pd.DataFrame({'responses':data}).to_json(r"synthetic_data_newest.json")


# %% --------------------------------------------------------------------------

# combine with pages
pages = df[['pages']]
outputs = pd.read_csv(r'data/synthetic_data.csv')
outputs.drop(columns='Unnamed: 0',inplace=True)
ready_data = pages.join(outputs)
#ready_data.to_csv(r'data/ready_data.csv')
ready_data.to_json(r'data/ready_data.json')
# %%
pd.read_json(r'data/ready_data.json')

# %%
#using gpt instruct - old
con = "### 202Using Recombinant DNA TechnologyThere are Concerns About the Use of Recombinant DNA Technology...There are ethical, financial and social issues associated with the use of recombinant DNA technology:AgricultureFarmers might plant only one type of transformed crop (this is called monoculture). This could make the whole crop vulnerable to the same disease because the plants are genetically identical. Environmentalists are also concerned about monocultures reducing biodiversity, as this could damage the environment.Some people are concerned about the possibility of 'superweeds' â€” weeds that are resistant to herbicides. These could occur if transformed crops interbreed with wild plants. There could then be an uncontrolled spread of recombinant DNA, with unknown consequences.Organic farmers can have their crops contaminated by wind-blown seeds from nearby genetically modified crops. This means they can't sell their crop as organic and may lose their income.IndustryAnti-globalisation activists oppose globalisation (e.g. the growth of large multinational companies at the expense of smaller ones). A few, large biotechnology companies control some forms of genetic engineering. As the use of this technology increases, these companies get bigger and more powerful. This may force smaller companies out of business, e.g. by making it harder for them to compete. Without proper labelling, some people think they won't have a choice about whether to consume food made using genetically engineered organisms.Some consumer markets, such as the EU, won't import GM foods and products. This can cause an economic loss to producers who have traditionally sold to those markets.MedicineCompanies who own genetic engineering technologies may limit the use of technologies that could be saving lives.Some people worry this technology could be used unethically, e.g. to make designer babies (babies that have characteristics chosen by their parents). This is currently illegal though.Recombinant DNA technology also creates ownership issues. Here are some examples:â€¢ There is some debate about who owns genetic material from humans once it has been removed from the body â€” the donor or the researcher. Some people argue that the individual holds the right to their own genetic information, however others argue that value is created by the researcher who uses it to develop a medicine or in diagnosis.â€¢ A small number of large corporations own patents to particular seeds. They can charge high prices, sometimes including a 'technology fee', and can require farmers to repurchase seeds each year.If non-GM crops are contaminated by GM crops, farmers can be sued for breaching the patent law....But Humanitarians Think it will Benefit PeopleRecombinant DNA technology has many potential humanitarian benefits:1) Agricultural crops could be produced that help reduce the risk of famine and malnutrition, e.g. drought-resistant crops for drought-prone areas.2) Transformed crops could be used to produce useful pharmaceutical products (e.g. vaccines) which could make drugs available to more people, e.g. in areas where refrigeration (usually needed for storing vaccines) isn't available.3) Mcdicines could be produced more chcaply, so more people can afford them.4) Recombinant DNA technology has the potential to be used in gene therapy to treat human diseases (see next page).>0U need to be able z- to balance the humanitarian : r h/neflts opposing views =- from environmentalists and ~ S ant|-9|obalisat,on activists r(see above) -Topic 8B â€” Genome Projects and Gene Technologies"
prompt = f"""Split the following text into key subsections and add informative titles for each subsection.
Each subsection must be in paragraph form and no information should be missing from the original text.
Do not remove information.
Subsections should be more than single sentences where possible.
Mark each title you create by adding the symbols "@@@" before each title.
An example subsection format is "@@@title  \n content", where you should add the subsection title and content.
This is the text: ###{con}###"""
os.environ["OPENAI_API_KEY"] = "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"
key = "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"
openai.api_key = key
client = OpenAI()
client.completions.create(
    model="gpt-3.5-turbo-instruct",
    prompt=prompt,
    max_tokens = 2500
                ).choices[0].text
# %%
#using gpt-3.5-turbo-1106 - longer context window and improved instruction folowing
import requests

con = '2Topic IA — Biological MoleculesCarbohydratesEven though there is, and has been, a huge variety of different organisms on Earth, they all share some biochemistry— for example, they all contain a few carbon-based compounds that interact in similar ways.Most Carbohydrates are Polymers1) Most carbohydrates (as well as proteins and nucleic acids) are polymers.2) Polymers are large, complex molecules composed of long chains of \'=4monomers joined together.3) Monomers are small, basic molecular units.4) Examples of monomers include monosaccharides, amino acids and nucleotides.monomer e.g. monosaccharide, amino acidhrpolymer e.g. carbohydrate, proteinCarbohydrates are Made from Monosaccharides1) All carbohydrates contain the elements C, H and O.2) The monomers that they\'re made from are monosaccharides, e.g. glucose, fructose and galactose.1) Glucose is a hexose sugar — a monosaccharide with six carbon atoms in each molecule.2) There are two types of glucose, alpha (a) and beta (|3)— they\'re isomers (molecules with the same molecular formula as each other, but with the atoms connected in a different way).3) You need to know the structures of both types of glucose for your exam— it\'s pretty easy because there\'s only one difference between the two:a-glucose moleculeP-glucose moleculeCH,OH i 2H\\/ h  \\  r \\Hn/  \\ ? H v  k  IHO c------c \\ OH /i, i \\ /H OH V^The two types of glucose have these groups reversedCH2OH\\A ~ \\rho/ VhI IH OHOH\nCondensation Reactions Join Monosaccharides Together1) A condensation reaction is when two molecules join together with the formation of a new chemical bond,and a water molecule is released when the bond is formed.2) Monosaccharides are joined together by condensation reactions.3) A glycosidic bond forms between the two monosaccharides as a molecule of water is released.4) A disaccharide is formed when two monosaccharides join together.ExampleTwo a-glucose molecules are joined together by a glycosidic bond to form maltose.HHOO. /Ha-glucoseo;h HO H O is removeda-glucoseHOHHHO5) Sucrose is a disaccharide formed from a condensation reaction between a glucose molecule and a fructose molecule.6) Lactose is another disaccharide formed from a glucose molecule and a galactose molecule.glycosidic bondO /H+ H,0lQ\'1maltoseOH1 Ml M M I I I I I n I I I I I^ If you\'re asked to show a tZ condensation reaction, don\'t ~ -- forget to put the water Iz molecule in as a product. r11111 1 1 1  n 1 1  n 1 / 11 ii 1 1Topic 1A — Biological Molecules\n3CarbohydratesHydrolysis Reactions Break Polymers Apart1) Polymers can be broken down into monomers by hydrolysis reactions.2) A hydrolysis reaction breaks the chemical bond between monomers using a water molecule. It\'s basically the opposite of a condensation reaction.3) For example, carbohydrates can be broken down into their constituent monosaccharides by hydrolysis reactions.PolymerAHydrolysis — the bond is broken by theaddition of a water molecule-OH HO--OHEven hydrolysis couldn\'t break this bond.Use the Benedict’s Test for SugarsSugar is a general term for monosaccharides and disaccharides. All sugars can be classified as reducing or non-reducing. The Benedict\'s test tests for sugars — it differs depending on the type of sugar you are testing for.1)2)3)Reducing sugars include all monosaccharides (e.g. glucose) and some disaccharides (e.g. maltose and lactose).You add Benedict\'s reagent (which is blue) to a sample and heat it in a water bath that\'s been brought to the boil.If the test\'s positive it will form a coloured precipitate (solid particles suspended in the solution).The colour of the precipitate changes from:blue-£> green>=>-yellowH>orange=4> brick redr Alwars us« an excess of r Z Benedict\'s solution — ;Z thls mak« sure that all ~ Z the sugar reacts. z1 1 1 " I I I I I I M M | | | V\\N4)The higher the concentration of reducing sugar, the further the colour change goes — you can use this to compare the amount of reducing sugar in different solutions. A more accurate way of doing this is to filter the solution and weigh the precipitate.C/1d£<UD1X1UZuDOLUIzO1) If the result of the reducing sugars test is negative, there could still be a non-reducing sugar present. To test for non-reducing sugars, like sucrose, first you have to break them down into monosaccharides.2) You do this by getting a new sample of the test solution, adding dilute hydrochloric acid and carefully heating it in a water bath that\'s been brought to the boil. You then neutralise it with sodium hydrogencarbonate. Then just carry out the Benedict\'s test as you would for a reducing sugar.3) If the test\'s positive it will form a coloured precipitate (as for the reducing sugars test). If the test\'s negative the solution will stay blue, which means it doesn\'t contain any sugar (either reducing or non-reducing).Topic 1A — Biological Molecules\n4CarbohydratesSo, you\'ve already looked at monosaccharides and disaccharides... now it\'s time to give polysaccharides some love.Polysaccharides are Loads of Sugars Joined TogetherA polysaccharide is formed when more than two monosaccharides are joined together by condensation reactions.\na-glucose a-glucose a-glucose a-glucose a-glucoseYou need to know about the relationship between the structure and function of three polysaccharides— starch, glycogen and cellulose.Starch is the Main Energy Storage Material in Plants\none alpha-glucose molecule1) Cells get energy from glucose. Plants store excess glucose as starch (when a plant needs more glucose for energy, it breaks down starch to release the glucose).2) Starch is a mixture of two polysaccharides of alpha-glucose — amylose and amylopectin:• Amylose — a long, unbranched chain of a-glucose. The angles of the glycosidic bonds give it a coiled structure, almost like a cylinder. This makes it compact, so it\'s really good for storage because you can fit more in to a small space.• Amylopectin — a long, branched chain of a-glucose. Its side branches allow the enzymes that break down the molecule to get at the glycosidic bonds easily.This means that the glucose can be released quickly. Amylopectin3) Starch is insoluble in water and doesn\'t affect water potential (see page 40), so it doesn\'t cause water to enter cells by osmosis, which would make them swell.This makes it good for storage.Use the Iodine Test for StarchIf you do any experiment on the digestion of starch and want to find out if any is left, you\'ll need the iodine test.just add iodine dissolved in potassium iodide solution to the test sample. If there is starch present, the sample changes from browny-orange to a dark, blue-black colour.Glycogen is the Main Energy Storage Material in AnimalsGlycogen1) Animal cells get energy from glucose too.But animals store excess glucose as glycogen— another polysaccharide of alpha-glucose.2) Its structure is very similar to amylopectin, except that it has loads more side branches coming off it. Loads of branches means that stored glucose can be released quickly, which is important for energy release in animals.3) It\'s also a very compact molecule, so it\'s good for storage.After throwing and fetching the ball no less than 312 times, Chappy and Stuart were finally out of glycogen.Topic 1A — Biological Molecules\n5CarbohydratesCellulose is the Major Component of Cell Walls in Plants1} Cellulose is made of long, unbranched chains of beta glucose.2) When beta-glucose molecules bond, they form straight cellulose chains.3) The cellulose chains are linked together by hydrogen bonds to form strong fibres called microfibrils. The strong fibres mean cellulose provides structural support for cells (e.g. in plant cell walls).one cellulose moleculeweak hydrogen bondsone beta-glucose moleculePractice QuestionsQ1 What is a polymer?Q2 Draw the structure of a-glucose.Q3 What type of bond holds monosaccharide molecules together in a polysaccharide?Q4 Name the two polysaccharides present in starch.Q5 Describe the iodine test for starch.Exam QuestionsQ1 Maltose is a sugar. Describe how a molecule of maltose is formed. [3 marks]Q2 Sugars can be classed as reducing or non-reducing. Describe the test used to identify a non-reducing sugar.Include the different results you would expect to see if the test was positive or negative. [5 marks]Q3 Read the following passage:Chitin is a structural polysaccharide, similar to cellulose in plants, that is found in the exoskeletons of insects and crustaceans, as well as in the cell walls of fungi. It is made up of chains of the monosaccharide N-acetylglucosaminc, which is derived from glucosc. The polysaccharidc chains arc long, unhranchcd and linked together by weak hydrogen bonds.Chitin can be broken down by enzymes callcd chitinascs, which catalyse hydrolysis reactions. Some organisms arc able to make their own chitinascs. Amongst these arc yeasts, such as Saccharomyces cerevisiae. In yeast reproduction, a newly formed yeast cell ‘buds off’ from the cell wall of its parent cell to become a new independent organism. This requires the separation of the cell wall of the new cell from the cell wall of the parent cell. Sacchammyces cerevisiae uses a chitinase for this purpose.Use information from the passage and your own knowledge to answer the following questions:a) Explain why chitin can be described as a polysaccharide (line 1). [1 mark]b) Chitin is similar to cellulose in plants (line I).Describe the ways in which cellulose and chitin are similar. [3 marks]c) Chitin can be broken down by enzymes called chitinases, which catalyse hydrolysis reactions (line 5).Explain how these hydrolysis reactions break down chitin. [2 marks]d) Some organisms arc able to make their own chitinascs (line 5 and 6).Explain how it would be bcncficial for plants to make and sccrctc chitinascs as a defence system. [4 marks]Starch — I thought that was just for shirt collars...Every coll in an organism is adapted to perform a function — you can always trace some of its features back to its function. Different cells even use the exact same molecules to do completely different things. Take glucose, for example— all plant cells use it to make cellulose, but they can also make starch from it if they need to store energy Smashing.Topic IA — Biological Molecules\n'
con1 = """based upon the central aim or mission of the business, but 
they are expressed in terms that provide a much clearer 
guide for management action or strategy.
Common corporate objectives
1 Profit maximisation
All the stakeholders in a business are working for reward. 
Profi ts are essential for rewarding investors in a business 
and for fi nancing further growth. Pro fi ts are necessary 
to persuade business owners Â â€“ or entrepreneurs Â â€“ to take 
risks. But what does â€˜pro fi t maximisationâ€™ really mean? In 
simple terms, it means producing at that level of output 
where the greatest positive di ff erence between total 
revenue and total costs is achieved.
Th e chief argument in support of this objective is that it 
seems rational to seek the maximum pro fi t available from 
a given venture. Not to maximise pro fi t, according to this 
objective, is seen as a missed opportunity. However, there 
are serious limitations with this corporate objective:
â–  The focus on high short-term profits may encourage 
competitors to enter the market and jeopardise the long-
term survival of the business.
â–  Many businesses seek to maximise sales in order to secure 
the greatest possible market share, rather than to maximise 
profits. The business would expect to make a target rate of 
profit from these sales.
â–  The owners of smaller businesses may be more concerned 
with ensuring that leisure time is safeguarded. The issues 
of independence and retaining control may assume greater 
significance than making higher profits.
â–  Most business analysts assess the performance of a 
business through return on capital employed rather than 
through total profit figures.
â–  Profit maximisation may well be the preferred objective of 
the owners and shareholders, but other stakeholders will 
give priority to other issues. Business managers cannot 
ignore these. Hence the growing concern over job security 
for the workforce or the environmental concerns of local 
residents may force profitable business decisions to be 
modified, yielding lower profit levels.
â–  In practice it is very di ff icult to assess whether the point 
of profit maximisation has been reached, and constant 
changes to prices or output to attempt to achieve it may 
well lead to negative consumer reactions.
2 Profit satisficing
Th is means aiming to achieve enough pro fi t to keep the 
owners happy but not aiming to work fl at out to earn as 
much pro fi t as possible. Th is objective is o ft en suggested 
as being common among owners of small businesses who 
wish to live comfortably but do not want to work longer 
and longer hours in order to earn even more pro fi t. Once a satisfactory level of pro fi t has been achieved, the owners 
consider that other aims take priority Â â€“ such as more 
leisure time.
3 Growth
Th e growth of a business Â â€“ usually measured in terms 
of sales or value of output Â â€“ has many potential bene fi ts 
for the managers and owners. Larger fi rms will be less 
likely to be taken over and should be able to bene fi t from 
economies of scale. Managers will be motivated by the 
desire to see the business achieve its full potential, from 
which they may gain higher salaries and fringe bene fi ts. 
It is also argued that a business that does not attempt to 
grow will cease to be competitive and, eventually, will lose 
its appeal to new investors. Business objectives based on 
growth do have limitations:
â–  expansion that is too rapid can lead to cash-flow problems
â–  sales growth might be achieved at the expense of lower 
profit margins
â–  larger businesses can experience diseconomies of scale
â–  u s i n g  p r o f i t s  t o  f i n a n c e  g r o w t h Â â€“ retained earnings Â â€“ can 
lead to lower short-term returns to shareholders
â–  g r o w t h  i n t o  n e w  b u s i n e s s  a r e a s  a n d  a c t i v i t i e s Â â€“ away from 
the firmâ€™s core activities Â â€“ can result in a loss of focus and 
direction for the whole organisation.
4 Increasing market share
Closely linked to overall growth of a business is the market 
share it enjoys within its main market. Although the two 
are usually related, it is possible for an expanding business 
to su ff er market share reductions if the market is growing 
at a faster rate than the business itself. Increasing market 
share indicates that the marketing mix of the business is 
proving to be more successful than that of its competitors. 
Bene fi ts resulting from having the highest market share Â â€“ 
being the brand leader Â â€“ include:
â–  retailers will be keen to stock and promote the best-selling 
brand
â–  p r o f i t  m a r g i n s  o ff ered to retailers may be lower than 
competing brands as the shops are so keen to stock it Â â€“ this 
leaves more profit for the producer
â–  eff ective promotional campaigns are o ft en based on â€˜buy 
our product with confidence Â â€“ it is the brand leaderâ€™.
5 Survival
Th is is likely to be the key objective of most new business 
start-ups. Th e high failure rate of new businesses means 
that to survive for the fi rst two years of trading is an 
important aim for entrepreneurs. Once the business 
has become fi rmly established, then other longer-term 
objectives can be established.42Cambridge International AS and A Level Business"""
prompt1 = f"""You will sort text into subsections and add informative titles for each subsection, without removing any information from the original text.
Each subsection must be in paragraph form and no information should be missing from the original text.
Make subsections as large as possible.
Mark each title you create by adding the symbols "@@@" before each title. For example, if i gave you this text: "17 1A.5 PROTEINS CHEMISTRY FOR BIOLOGISTS\nHYDROGEN BONDS\nY ou were introduced to hydrogen bonds in Section 1A.1. These same bonds are essential in protein \nstructures. In amino acids, tiny negative charges are present on the oxygen of  the carboxyl groups and tiny positive charges are present on the hydrogen atoms of  the amino groups. When these charged groups are close to each other, the opposite charges attract, forming a hydrogen bond. Hydrogen bonds are weak but, potentially, they can be made between any two amino acids in the correct position, so there are many of  them holding the protein together very firmly. They are very important in the folding and coiling of  polypeptide chains (see fig C). Hydrogen bonds break easily and reform if  pH or temperature conditions change.\nDISULFIDE BONDS\nDisulfide bonds form when two cysteine molecules are close together in the structure of  a polypeptide (see fig C). An oxidation reaction occurs between the two sulfur-containing groups, resulting in a strong covalent bond known as a disulfide bond. These disulfide bonds are much stronger than hydrogen bonds but they happen much less often. They are important for holding the folded polypeptide chains in place.\nhydrogen bondhydrogen bond\ndisulﬁde bonddisulﬁde bond\nα-helixα-helixα-helixα-helixβ-pleated sheetβ-pleated sheetβ-pleated sheetβ-pleated sheet\n▲ fig C  Hydr ogen bonds and disulfide bonds maintain the shape of protein molecules and this determines  \ntheir function." I would expect you to give me an output like this: "@@@HYDROGEN BONDS
You were introduced to hydrogen bonds in Section 1A.1. These same bonds are essential in protein structures. In amino acids, tiny negative charges are present on the oxygen of the carboxyl groups and tiny positive charges are present on the hydrogen atoms of the amino groups. When these charged groups are close to each other, the opposite charges attract, forming a hydrogen bond. Hydrogen bonds are weak but, potentially, they can be made between any two amino acids in the correct position, so there are many of them holding the protein together very firmly. They are very important in the folding and coiling of polypeptide chains (see fig C). Hydrogen bonds break easily and reform if pH or temperature conditions change.

@@@DISULFIDE BONDS
Disulfide bonds form when two cysteine molecules are close together in the structure of a polypeptide (see fig C). An oxidation reaction occurs between the two sulfur-containing groups, resulting in a strong covalent bond known as a disulfide bond. These disulfide bonds are much stronger than hydrogen bonds but they happen much less often. They are important for holding the folded polypeptide chains in place.
hydrogen bondhydrogen bond
disulﬁde bonddisulﬁde bond
α-helixα-helixα-helixα-helixβ-pleated sheetβ-pleated sheetβ-pleated sheetβ-pleated sheet
▲ fig C Hydrogen bonds and disulfide bonds maintain the shape of protein molecules and this determines their function." Do you understand the task?"""
response1 = 'Yes, I understand the task. Please provide the text you would like me to sort into subsections and add informative titles for each subsection, without removing information.'
prompt2 = f"This is the text: {con}"

url = "https://api.openai.com/v1/chat/completions"
headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer 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"
}

data = {
    #"model": "gpt-3.5-turbo-1106",
    "model": "gpt-4-1106-preview",
    "messages": [
        {
            "role": "system",
            "content": "You are a helpful assistant who will follow the instructions of the user"
        },
        {
            "role": "user",
            "content": f"{prompt1}"
        },
        {
            "role": "assistant",
            "content": f"{response1}"
        },
        {
            "role": "user",
            "content": f"{prompt2}"
        }
    ],
    'temperature':1
}

response = requests.post(url, headers=headers, json=data)

# Print the response content
res = (response.json())
message = res['choices'][0]['message']['content']

# %%
from transformers import AutoTokenizer
model_id = "TheBloke/Mistral-7B-v0.1-GPTQ"

tokenizer = AutoTokenizer.from_pretrained(
    model_id, use_fast=False, add_eos_token=True
)
# %%
lens = []
for i,j in enumerate(df['pages']): 
    if len(tokenizer(j)['input_ids']) ==1428:
        print(i)


# %%
def test():
    res = []
    try:
        for i in range(1,100):
    
                
            print(i)
            res.append(i/(i-10))
    except:
        None
    return res
hi = test()