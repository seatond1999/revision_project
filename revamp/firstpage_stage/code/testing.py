
# %%
import pandas as pd
contents = pd.DataFrame({'chapter_titles': ['1. Quantities and units', '2. Practical skills', '3. Rectilinear motion',
                            '4. Momentum', '5. Forces', '6. Work, energy and power', '7. Charge and current',
                            '8. Potential difference, electromotive force ...', '9. Current–potential difference relationships',
                            '10. Resistance and resistivity', '11. Internal resistance, series and parallel ...',
                            '12. Fluids', '13. Solid materials', '14. Nature of waves', '15. Transmission and reflection of waves',
                            '16. Superposition of waves', '17. Particle nature of light', '18. Maths in physics',
                            '19. Preparing for the exams', 'Index'],
        'page_number': [1, 10, 25, 44, 55, 73, 90, 102, 116, 128, 145, 173, 188, 210, 221, 247, 274, 296, 310, 323],
        'corrected_page_number': [10, 19, 34, 53, 64, 82, 99, 111, 125, 137, 154, 182, 197, 219, 230, 256, 283, 305, 319, 332]})

# -----------------------------------------------------------------------------

# %%
def split_chosen_chapter(chosen_chapter,contents):
    #self.set_adapter('causal')
    #self.model = self.model.to("cuda:0")
    chosen_chapter_index = contents[contents['chapter_titles'] == chosen_chapter].index[0] #gets row of chosen chapter in contents df
    sub_titles = []
    text = []
    for i in range(contents.iloc[chosen_chapter_index,2],contents.iloc[chosen_chapter_index+1,2]): #between the page numbers of chosen chapter and next chapter
        print(i)

split_chosen_chapter('1. Quantities and units',contents)
# %%
