import numpy as np


# def stable_matching(matrix):
#     # (num natural language, num image)
#     num_nl, num_img = matrix.shape
#     # unmatched_nl = np.ones(num_nl)
#     # unmatched_img = np.ones(num_img)
    
#     # matching_list = [-1] * num_nl
#     matched_list = []
#     # propose
#     c = 0
#     while len(matched_list) != num_nl:
#         img_preference_of_nl = np.argmax(matrix, axis=1)
#         bool_matrix_of_img_preference_per_nl = np.zeros_like(matrix)
#         bool_matrix_of_img_preference_per_nl[np.arange(num_nl), img_preference_of_nl] = 1

#         nl_preference_of_img = np.argmax(matrix, axis=0)
#         bool_matrix_of_nl_preference_per_img = np.zeros_like(matrix)
#         bool_matrix_of_nl_preference_per_img[nl_preference_of_img, np.arange(num_img)] = 1
#         final_preference_matrix = bool_matrix_of_nl_preference_per_img * bool_matrix_of_img_preference_per_nl * (matrix != -1)
#         preference_coord = np.transpose(np.nonzero(final_preference_matrix))
#         matched_list += preference_coord.tolist()
#         matrix[final_preference_matrix.sum(axis=1)] = -1
#         matrix[:, final_preference_matrix.sum(axis=0)] = -1

#     return sorted(matched_list)


def stableMatching(matrix):
    # Initially, all n men are unmarried
    num_nl, num_img = matrix.shape
    menPreferences = (-matrix).argsort(axis=1).tolist()
    womenPreferences = np.transpose((-matrix).argsort(axis=0)).tolist()
    unmarriedMen = list(range(num_nl))
    # None of the men has a spouse yet, we denote this by the value None
    manSpouse = [None] * num_nl                      
    # None of the women has a spouse yet, we denote this by the value None
    womanSpouse = [None] * num_img                      
    # Each man made 0 proposals, which means that 
    # his next proposal will be to the woman number 0 in his list
    nextManChoice = [0] * num_nl                       
    
    # While there exists at least one unmarried man:
    while unmarriedMen:
        # Pick an arbitrary unmarried man
        he = unmarriedMen[0]                      
        # Store his ranking in this variable for convenience
        hisPreferences = menPreferences[he]       
        # Find a woman to propose to
        she = hisPreferences[nextManChoice[he]] 
        # Store her ranking in this variable for convenience
        herPreferences = womenPreferences[she]
        # Find the present husband of the selected woman (it might be None)
        currentHusband = womanSpouse[she]
       
        
        # Now "he" proposes to "she". 
        # Decide whether "she" accepts, and update the following fields
        # 1. manSpouse
        # 2. womanSpouse
        # 3. unmarriedMen
        # 4. nextManChoice
        if currentHusband == None:
            #No Husband case
            #"She" accepts any proposal
            womanSpouse[she] = he
            manSpouse[he] = she
            #"His" nextchoice is the next woman
            #in the hisPreferences list
            nextManChoice[he] = nextManChoice[he] + 1
            #Delete "him" from the 
            #Unmarried list
            unmarriedMen.pop(0)
        else:
            #Husband exists
            #Check the preferences of the 
            #current husband and that of the proposed man's
            currentIndex = herPreferences.index(currentHusband)
            hisIndex = herPreferences.index(he)
            #Accept the proposal if 
            #"he" has higher preference in the herPreference list
            if currentIndex > hisIndex:
                #New stable match is found for "her"
                womanSpouse[she] = he
                manSpouse[he] = she
                nextManChoice[he] = nextManChoice[he] + 1
                #Pop the newly wed husband
                unmarriedMen.pop(0)
                #Now the previous husband is unmarried add
                #him to the unmarried list
                unmarriedMen.insert(0,currentHusband)
            else:
                nextManChoice[he] = nextManChoice[he] + 1
    return manSpouse
    

if __name__ == '__main__':
    a = np.array([
        [90, 80, 70],
        [60, 50, 40],
        [80, 90, 10]
    ])
    print(stableMatching(a))