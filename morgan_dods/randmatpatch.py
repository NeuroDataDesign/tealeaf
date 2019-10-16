def paramsRandMatImagePatch(ih, iw, ph_min, ph_max, pw_min, pw_max, num_patches):
    # generate random patch list (height, width, (top_left_row, top_left_col) )
    # param ih: image height (integer)
    # param iw: image width (integer)
    # param ph_min: patch height minimum (integer)
    # param ph_max: patch height maximum (integer)
    # param pw_min: patch width minimum (integer)
    # param pw_max: patch width maximum (integer)
    # param num_patches: desired number of patches (integer)
    # return : list of patches
    
    #Initialize patch descriptors
    deltah = 0
    deltaw = 0
    topLeft = (0,0)
    bottomRight = (0,0)
    
    #generate num_patches patches by randomly choosing 
    #a height, width, and top left coordinate
    patches = []
    for i in range(num_patches):
        # generate random patch height and width
        deltah = random.randint(ph_min, ph_max)
        deltaw = random.randint(wh_min, pw_max)
        
        # generate random patch top left corner by 
        # restricting based on patch height and width
        ih_max = ih - deltah - 1 # original adds 1, not sure why
        iw_max = iw - deltaw - 1 
        #stored as tuple instead of single value
        topLeft = (random.randint(0,ih_max), random.randint(0,iw_max)) 
        
        # store height, width, and top left corner index in patches list
        patches.append((deltah, deltaw, tuple(topLeft))) # do I need to deep copy? 
        
    return patches

def randMatImagePatch(patches):
    # Generate list containing patches and weights 
    # [((height1, width1, (top_left_row1, top_left_col1) ), weight1, ...]
    # param patches: list of patches
    # return : features to try list
    
    # original only provides pixel index and weight of 1
    # not sure why this is a separate function? 
    # and why height and width are not included?
    # should I be working with arrays?
    
    return features_to_try = [[patch, 1] for i in patches] 
    
        