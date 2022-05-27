# import the necessary packages
from skimage.segmentation import slic
import numpy as np
from PIL import Image, ImageFilter



# define set values (for now, need to be checked for the used INN)
avg_value = 117
img_shape = (224, 224)

def create_segments(image, n_segs):    
    segment_list = []
    # loop over the number of segments
    for numSegments in n_segs:     #(50, 150): #replace with (15, 50, 80)
        # apply SLIC and extract (approximately) the supplied number of segments
        segments = slic(image, n_segments = numSegments, sigma = 5, start_label=1)        
        segment_list.append(segments)
    
    return segment_list

def extract_patch(image, mask, bg_mode):
    # mask image and add grey background
    if bg_mode == 'patch':
        # get crop locations and crop image
        ones = np.where(mask == 1)
        h1, h2, w1, w2 = ones[0].min(), ones[0].max(), ones[1].min(), ones[1].max()
        cropped_img = image.crop((w1, h1, w2, h2))
        
        # resize cropped image to forward through INN
        resized_img = cropped_img.resize((img_shape))
        
        patch = (w1, h1, w2, h2)
    else:
        if bg_mode == 'grey':
            base_img = Image.new('RGB', image.size, color=(117,117,117))
        elif bg_mode == 'blurred':
            base_img = image.filter(ImageFilter.GaussianBlur(5))
        else: 
            raise Exception("background mode should be either 'grey' or 'blurred'.")

        mask_img = Image.fromarray(np.uint8(255*mask))
        segment_img = Image.composite(image, base_img, mask_img)
        
        # get crop locations and crop image
        ones = np.where(mask == 1)
        h1, h2, w1, w2 = ones[0].min(), ones[0].max(), ones[1].min(), ones[1].max()
        cropped_img = segment_img.crop((w1, h1, w2, h2))
        
        # resize cropped image to forward through INN
        resized_img = cropped_img.resize((img_shape))
        
        patch = (w1, h1, w2, h2)

    return resized_img, patch

def return_superpixels(image, n_segs = [15, 50, 80], bg_mode = 'grey'):
    segment_list = create_segments(image, n_segs = n_segs)
    
    unique_masks = []
    for segments in segment_list:
        param_masks = []
        for s in range(segments.max() + 1): # +1 because it starts labeling at 0
            mask = (segments == s).astype(float)
            if np.mean(mask) > 0.001:
                unique = True
                for seen_mask in unique_masks:
                    jaccard = np.sum(seen_mask * mask) / np.sum((seen_mask + mask) > 0)
                    if jaccard > 0.5:
                        unique = False
                        break
                if unique:
                    param_masks.append(mask)
        unique_masks.extend(param_masks)

    superpixels, patches = [], []
    while unique_masks:
        superpixel, patch = extract_patch(image, unique_masks.pop(), bg_mode)
        superpixels.append(superpixel)
        patches.append(patch)
        
    return superpixels, patches