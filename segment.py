from PIL import Image, ImageDraw
import os
import cv2
import numpy as np
import math
from scipy.stats import linregress
from matplotlib import pyplot as plt
import scipy
import pybobyqa
from typing import Tuple
from collections import namedtuple

multiplier_ver = 8.25
multiplier_hor = 1.75
DeslantRes = namedtuple('DeslantRes', 'img, shear_val, candidates')
Candidate = namedtuple('Candidate', 'shear_val, score')

def _get_shear_vals(lower_bound: float,
                    upper_bound: float,
                    step: float) -> Tuple[float]:
    """Compute shear values in given range."""
    return tuple(np.arange(lower_bound, upper_bound + step, step))

def _shear_img(img: np.ndarray,
               s: float, bg_color: int,
               interpolation=cv2.INTER_NEAREST) -> np.ndarray:
    """Shears image by given shear value."""
    h, w = img.shape
    offset = h * s
    w = w + int(abs(offset))
    tx = max(-offset, 0)

    shear_transform = np.asarray([[1, s, tx], [0, 1, 0]], dtype=float)
    img_sheared = cv2.warpAffine(img, shear_transform, (w, h), flags=interpolation, borderValue=bg_color)

    return img_sheared

def _compute_score(img_binary: np.ndarray, s: float) -> float:
    """Compute score, with higher score values corresponding to more and longer vertical lines."""
    img_sheared = _shear_img(img_binary, s, 0)
    h = img_sheared.shape[0]

    img_sheared_mask = img_sheared > 0
    first_fg_px = np.argmax(img_sheared_mask, axis=0)
    last_fg_px = h - np.argmax(img_sheared_mask[::-1], axis=0)
    num_fg_px = np.sum(img_sheared_mask, axis=0)

    dist_fg_px = last_fg_px - first_fg_px
    col_mask = np.bitwise_and(num_fg_px > 0, dist_fg_px == num_fg_px)
    masked_dist_fg_px = dist_fg_px[col_mask]

    score = sum(masked_dist_fg_px ** 2)
    return score

def block_image(im_file):
    im = cv2.imread(im_file,1)     

    rgb_planes = cv2.split(im)

    result_planes = []
    result_norm_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        result_planes.append(diff_img)
        result_norm_planes.append(norm_img)
    
    im = cv2.merge(result_norm_planes)

    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    
    return im

def preprocess(path_to_image):    
    while os.stat(path_to_image).st_size > 3000000:
        im = Image.open(path_to_image)
        width, height = im.size
        im = im.resize(int(round(width * .75)), int(round(height * .75)))
        im.save(path_to_image)
        im.close()

def GCD(x, y):
    while(y):
       x, y = y, x % y
    return abs(x)

def layout_analyze(data):
    # img = pooled
    # data = np.asarray(img)
    blocks = []
    coordinates = []
    binarization_quantile = 0.1
    bin_thresh = np.quantile(data, binarization_quantile)
    print("Dynamic binarization threshold = "+str(bin_thresh))

    for y in range(len(data)):

        for x in range(len(data[0])):  # TODO keep in mind the last window
              # print('ERROR', y, x)
            
            # if data[y][x] <= 175:  # TODO Find range of pixel values for text           
            
            if data[y][x] <= bin_thresh:
                data[y][x] = 0
            else:
                data[y][x] = 255

    # tmp_img = Image.fromarray(data)
    # tmp_img.show()

    h, w = len(data), len(data[0])
    print("height", h, "width", w)

    count = GCD(h, w)
    print("gcd",count)
    wy = len(data)//count
    wx = len(data[0])//count
    conv = []
    density = []
    # conv = [[False for i in range(count)] for j in range(count)]
    for i in range(count):
        conv.append([])
        for j in range(count):

            pixels = 0
            total = 0
            for y in range(i*wy, (i+1)*wy):
                for x in range(j*wx, (j+1)*wx):
                    if data[y][x] == 0:
                        pixels += 1
                    total += 1
            density.append(pixels/total)
            # print(i, j, pixels/total)
            # if pixels/total >= 0.004:
            conv[-1].append([(j*wx, i*wy, (j+1)*wx, (i+1)*wy), pixels/total])
    # print(windows)
    mean = np.mean(density)
    std = np.std(density)

    img = Image.fromarray(data)
    img = img.convert('RGB')
    draw = ImageDraw.Draw(img, "RGBA")

    visited = set()
    ignore = set()
    threshold = 100000000  # for running sequence of windows
    # print(mean, std, mean+std**2, mean+0.2*std)
    criterion = mean+std**2

    for y in range(len(conv)):
        for x in range(len(conv[0])):
            if (y, x) in visited:
                continue

            px = x
            count = 0
            # print(conv[y][px])
            while px < len(conv[0]) and conv[y][px][1] >= criterion:
                visited.add((y, px))
                px += 1
                count += 1

            if count > threshold:
                for i in range(x, px):
                    ignore.add((y, i))

    visited = set()
    for x in range(len(conv[0])):
        for y in range(len(conv)):
            if (y, x) in visited:
                continue

            py = y
            count = 0
            while py < len(conv) and conv[py][x][1] >= criterion:
                visited.add((py, x))
                py += 1
                count += 1

            if count > threshold:
                for i in range(y, py):
                    ignore.add((i, x))

    for y in range(len(conv)):
        for x in range(len(conv[0])):
            if conv[y][x][1] >= criterion and (y, x) not in ignore:
                # draw.rectangle(conv[y][x][0], outline = 'blue', fill=(255, 0, 0, 30))
                continue

    ver_count = []

    for x in range(len(conv[0])):
        cur = 0
        for y in range(len(conv)):
            if conv[y][x][1] >= criterion and (y, x) not in ignore:
                cur += 1
        # print(cur)
        ver_count.append(cur)

    # print(conv)
    # print("ver_count", ver_count)

    def moving_average(x, w):        
        return np.convolve(x, np.ones(w), 'valid') / w

    mov_ver_count = moving_average(ver_count, 3)
    # print("mov_ver_count", mov_ver_count)
    dif = []

    #what is supposed to happen here when there is only one element in mov_ver_count?
    """if len(mov_ver_count) == 1:
        dif = mov_ver_count
    else:"""
    for i in range(1, len(mov_ver_count)):
        dif.append(abs(mov_ver_count[i]-mov_ver_count[i-1]))

    moving_window = 20
    dif = moving_average(dif, moving_window)
    avg1 = [np.mean(dif)+multiplier_ver*np.std(dif)/(math.sqrt(len(dif)))
           for i in range(len(dif))]

    regress = linregress([i for i in range(len(dif))], dif)

    avg = [0.9*regress.slope*i+1.57*regress.intercept for i in range(len(dif))]
    left = right = 0
    pad = 5
    bounds = []
    min_size = 5
    while right < len(dif):
        while right < len(dif) and dif[right] > avg[right]:
            right += 1
        left = right
        while right < len(dif) and dif[right] <= avg[right]:
            right += 1
        if right-left > min_size:
            bounds.append([max(left+1*moving_window//4-pad,0), min(right+3*moving_window//4+pad, len(dif))])
    plt.plot(avg)
    plt.plot(avg1)
    plt.plot(dif)
    # plt.show()



    # print(bounds)
    # for b in bounds:
    #     draw.rectangle((b[0]*wx, 0, b[1]*wx, h), outline = 'blue', fill=(0, 255, 0, 30))
    # img.show()

    # for k in range(count):
    #     for l in range(count):
    #         draw.rectangle((wx*k, wy*l, wx*(k+1), wy*(l+1)), outline = 'black', fill=(255, 0, 0, 30))
    # img.show()

    for b in bounds:
        hor_count = []
        for y in range(len(conv)):
            cur = 0
            for x in range(b[0], min(b[1]+1, len(conv[0]))):
                if conv[y][x][1] >= criterion and (y, x) not in ignore:
                    cur += 1
            hor_count.append(cur)
        # mov_hor_count = moving_average(hor_count, 1)
        win_size = 5
        new_hor_count = []
        for i in range(len(hor_count)//win_size+1):
            cur = 0
            for j in range(win_size*i, min(win_size*(i+1), len(hor_count))):
                cur += hor_count[j]
            new_hor_count.append(cur)
        hor_count = new_hor_count

        dif = []

        # for i in range(1,len(mov_hor_count)):
        #     dif.append(abs(mov_hor_count[i]-mov_hor_count[i-1]))
        if len(hor_count) == 1:
            dif = hor_count
        else:
            for i in range(1, len(hor_count)):
                dif.append(abs(hor_count[i]-hor_count[i-1]))
        dif = moving_average(dif, 2)
        avg = [np.mean(dif)+multiplier_hor*np.std(dif)/math.sqrt(len(dif)) for i in range(len(dif))]

        regress = linregress([i for i in range(len(hor_count))], hor_count)

        avg = [regress.slope*i+0.5*regress.intercept for i in range(len(hor_count))]

        plt.plot(avg)
        plt.plot(hor_count)
        # # plt.plot(dif)
        # plt.show()

        left = right = 0
        pad = 0
        hor_bounds = []

        while right < len(hor_count):
            val = 0
            while right < len(hor_count) and hor_count[right] < avg[right]:
                right += 1
            left = right
            tolerate = 1
            while right < len(hor_count) and (hor_count[right] >= avg[right] or tolerate > 0):
                val += hor_count[right]
                if hor_count[right] > avg[right]:
                    tolerate -= 1
                right += 1
            if right-left > 2:
                # print(val, criterion*(right-left)*(min(b[1]+1,len(conv[0])-b[0])))
                # print(val, criterion*(right-left)*(min(b[1]+1,len(conv[0]))-b[0])*win_size, win_size*(right-left)*(min(b[1]+1,len(conv[0]))-b[0]), b, left, right)

                if val >= criterion*(right-left)*(min(b[1]+1, len(conv[0]))-b[0])*win_size*2 and (right-left)*(min(b[1]+1, len(conv[0]))-b[0])*win_size*wy*wx/(h*w) > 0.002:
                    hor_bounds.append([left-pad, right+pad])

        for hb in hor_bounds:
            x0 = b[0]*wx
            y0 = hb[0]*wy*win_size
            x1 = b[1]*wx
            y1 = hb[1]*wy*win_size
            # print(hb[0], hb[1], (x1-x0)/(y1-y0))
            if 1/6 < (x1-x0)/(y1-y0) < 6:
                
                #boxes.append([(x0, h-y0), (x1,h-y0),(x0,h-y1),(x1,h-y1)])
                # print('B',boxes)
                blocks.append(img.crop((x0, y0, x1, y1)))
                #blocks.append(img.crop((x0/w, y0/h, x1/w, y1/h)))
                # coordinates.append([x0/w, y0/h, x1/w-x0/w, y1/h-y0/h])
                #coordinates.append([x0/w, y0/h, x1/w, y1/h])

                #TODO these are currently going to be coordinates from the *resized* (960 x 1280) version of the image
                #I made this choice because we can't guarantee that the aspect ratio of the images are going to be preserved when they're resized
                #this needs to be taken into account if/when lines are being regenerated for use in model training
                #the exception to this rule is 239746 (or any other volume for which layout analysis has been done manually), since they won't be resized
                coordinates.append([x0, y0, x1, y1])
                draw.rectangle((x0, y0, x1, y1), outline= 'blue', fill=(0, 255, 0, 30))
    #img.show()
    return blocks, coordinates

def filter_blocks(blocks, coordinates):
    block_areas = []
    total_area = 0
    for block in blocks:
        block_areas.append(block.width * block.height)              
    for area in block_areas:
        total_area += area
    if len(block_areas) > 0:   
        avg_area = total_area / len(block_areas)
    else:
        return None, None
    entry_blocks = []
    entry_coords = []   
    for index, block in enumerate(blocks):        
        if block.width * block.height > .25 * avg_area:
            entry_blocks.append(block)
            entry_coords.append(coordinates[index])
    return entry_blocks, entry_coords

def gray_and_rotate(block):
    '''
    Function to convert input image into grayscale and rotate it appropriately
    Params:
        - block (Image): a block of text from the image
        - data (numpy array): rotated, grayscaled image as a numpy array
        - image_file (PIL Image): rotated, grayscaled image as an image object
        - orig-image (PIL Image): rotated, non-grayscaled image as an image object
    Returns:
        - data (numpy array): rotated and greyscaled image as a numpy array
        - image_file (PIL Image): rotated and greyscaled image object
        - orig_image (PIL Image): the original image that has not been rotated nor greyscaled
    '''
    # image_file = Image.open(filename)
    
    image_file = block    
    orig_image = block
    image_file = image_file.convert('L')
    
    rotate_degree = get_degree(image_file)
    image_file = image_file.rotate(rotate_degree, fillcolor=127)
    orig_image = orig_image.rotate(rotate_degree, fillcolor=127)
    
    data = np.asarray(image_file)
    return data, image_file, orig_image

def get_degree(orig_image_gray):
    '''
    Function to compute the degree needed to rotate the image
    Params:
        - orig_image_gray (PIL Image): the original grayscaled, non-rotated image
    Returns:
        - minDeg (int): the degree needed to rotate the image
    '''
    dict = {}
    orig_image_gray = orig_image_gray.resize((orig_image_gray.size[0]//3, orig_image_gray.size[1]//3))
    for i in range(-10, 10):
        # image_file = orig_image_gray
        image_file = orig_image_gray.rotate(i, fillcolor=1)
        data = np.asarray(image_file)
        pixel_counts = np.sum(data, axis=1, keepdims=True)
        array = [] #TODO can convert to np.sum for performance boost
        for val in pixel_counts:  # flatten the numpy array
            array.append(data.shape[1] - val[0])
        crop_pixels = scipy.signal.find_peaks(array, prominence=3000)[0]
        peaks = []
        for pixel in crop_pixels:
            peaks.append(array[pixel])
        peaks_dif = []
        for j in range(len(crop_pixels)-1):
            peaks_dif.append(crop_pixels[j+1]-crop_pixels[j])

        pixel_counts = np.sum(data, axis=1, keepdims=True)
        array = []
        for val in pixel_counts:  # flatten the numpy array
            array.append(val[0])
        crop_pixels = scipy.signal.find_peaks(array, prominence=3000)[0]
        troughs = []
        for pixel in crop_pixels:
            troughs.append(data.shape[1] - array[pixel])
        troughs_dif = []
        for j in range(len(crop_pixels)-1):
            troughs_dif.append(crop_pixels[j + 1] - crop_pixels[j])
        if len(peaks_dif) <= 1 or len(troughs_dif) <= 1:
            #set to arbitrarily large values
            dif = 10000
            std_vals = 10000
        else:
            dif = (np.std(peaks_dif)+np.std(troughs_dif))
            std_vals = (np.std(peaks) + np.std(troughs))

        #formula
        dict[i] = (std_vals)*(dif**3) / (((np.mean(peaks) - np.mean(troughs))**3)*(len(peaks)**4)*(len(troughs)**4))

    minDeg = 0
    minVal = 99999
    for degree in dict:
        if abs(dict[degree]) < minVal:
            minVal = abs(dict[degree])
            minDeg = degree
    return minDeg

def find_pixels(data, prominance):
    '''
    Find pixel boundaries to crop the images into lines using pixel-histogram analysis
    Params:
        - data (numpy array): rotated image as a numpy array
        - prominance (float): the prominance parameter to define the find_peaks method
    @:return crop_pixels array of indices indicate where to crop the image
    '''
    pixel_counts = np.sum(data, axis=1, keepdims=True) #sum pixels along the horizontal axis
    array = []
    for val in pixel_counts: #flatten the numpy array
        array.append(val[0])
    crop_pixels = scipy.signal.find_peaks(array, prominence=prominance)[0] #find pixel boundaries
    return crop_pixels

def data_segmentation(data, crop_pixels, file_name, image_file, coords, start):
    '''
    Function to crop input image by lines and output cropped images as specified by pixel boundaries
    The resulting images will be saved to disk in the segmented directory
    Params:
        - data (numpy array): rotated image as a numpy array
        - crop_pixels array (list): of indices indicate where to crop the image
        - file_name (str): name of the root file
        - image_file (PIL Image): the gray-scale version of the image file object to be cropped        
        - block_id (int): id of the current text block
        - coords (list): coordinates of block in original image
    Returns:
        - number of saved segments and a _ of their dimensions
    '''
    #establising initial boundaries
    top = 0
    left = 0
    right = data.shape[1]
    bottom = 0
    id = start #for name generation
    index = 0

    count = 0

    segment_coords = []

    #binarizing with hard threshold

    # image_file = image_file.point(lambda p: 255 if p > 200 else p) #threshold = 110
    # image_file.show()
    # image_file = image_file.point(lambda p: 255 if p > 200 else p)
    # print('BLOCK ID', block_id)

    if not os.path.exists("segmented"):
        os.mkdir("segmented")
    
    while index < len(crop_pixels): #iteratively crop the image with pixel boundaries
        # left = bottom
        top = bottom
        bottom = crop_pixels[index]

        # remove strips that are too small
        # this will likely eventually need to be tweaked for each volume/group of volumes
        # while bottom - left < 45:
        while bottom - top < 10:
            index += 1
            if index >= len(crop_pixels):
                return count, segment_coords
            # left = bottom
            top = bottom
            bottom = crop_pixels[index]


        # tmp_img = image_file.crop((top, left, right, bottom))        
        tmp_img = image_file.crop((left, top, right, bottom))

        #deslanting
        tmp_img = deslant_img(np.array(tmp_img))
        tmp_img = Image.fromarray(tmp_img.img)

        #if True:
        if evaluateImage(image_file):
            if id < 10:
                idx = '0'+str(id)
            else:
                idx = str(id)            
            tmp_img.save("segmented\\" + file_name + '-' + idx +'.jpg') #save output image
            segment_coords.append([left + coords[0], top + coords[1], right + coords[0], bottom + coords[1]])
            count += 1
            id+=1

        index+=1
    # image_file = image_file.crop((top, bottom, right, data.shape[0]))
    image_file = image_file.crop((left, bottom, right, data.shape[0]))

    image_file = deslant_img(np.array(image_file))
    image_file = Image.fromarray(image_file.img)


    #if True:
    if evaluateImage(image_file):
        if id < 10:
            idx = '0'+str(id)
        else:
            idx = str(id)       
        if image_file.size[1] > 9:
            image_file.save("segmented\\" + file_name + '-' + idx + '.jpg')  # save output image
            segment_coords.append([left + coords[0], bottom + coords[1], right + coords[0], data.shape[0] + coords[1]])
            count += 1
    
    return count, segment_coords

def deslant_img(img: np.ndarray,               
                optim_algo: 'str' = 'grid',
                lower_bound: float = -2,
                upper_bound: float = 2,
                num_steps: int = 20,
                bg_color=255) -> DeslantRes:
    """
    Deslants the image by applying a shear transform.

    The function searches for a shear transform that yields many long connected vertical lines.

    Args:
        img: The image to be deslanted with text in black and background in white.
        optim_algo: Specify optimization algorithm searching for the best scoring shear value:
            'grid': Search on grid defined by the bounds and the number of steps.
            'powell': Apply the derivative-free BOBYQA optimizer from Powell within given bounds.
        lower_bound: Lower bound of shear values to be considered by optimizer.
        upper_bound: Upper bound of shear values to be considered by optimizer.
        num_steps: Number of grid points if optim_algo is 'grid'.
        bg_color: Color that is used to fill the gaps of the returned sheared image.

    Returns:
        Object of DeslantRes, holding the deslanted image and (only for optim_algo 'grid') the candidates
        with shear value and score.
    """

    assert img.ndim == 2
    assert img.dtype == np.uint8
    assert optim_algo in ['grid', 'powell']
    assert lower_bound < upper_bound

    # apply Otsu's threshold method to inverted input image
    img_binary = cv2.threshold(255 - img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1] // 255

    # variables to be set by optimization method
    best_shear_val = None
    candidates = None

    # compute scores on grid points
    if optim_algo == 'grid':
        step = (upper_bound - lower_bound) / num_steps
        shear_vals = _get_shear_vals(lower_bound, upper_bound, step)
        candidates = [Candidate(s, _compute_score(img_binary, s)) for s in shear_vals]
        best_shear_val = sorted(candidates, key=lambda c: c.score, reverse=True)[0].shear_val

    # use Powell's derivative-free optimization method to find best scoring shear value
    elif optim_algo == 'powell':
        bounds = [[lower_bound], [upper_bound]]
        s0 = [(lower_bound + upper_bound) / 2]

        # minimize the negative score
        def obj_fun(s):
            return -_compute_score(img_binary, s)

        # the heuristic to find a global minimum is used, as the negative score contains many small local minima
        res = pybobyqa.solve(obj_fun, x0=s0, bounds=bounds, seek_global_minimum=True)
        best_shear_val = res.x[0]

    res_img = _shear_img(img, best_shear_val, bg_color, cv2.INTER_LINEAR)
    return DeslantRes(res_img, best_shear_val, candidates)

def evaluateImage(image_file):
    '''
    Function to analyze central density of the segmented strip of text
    If the density is too low, then it is a false positive segmentation and should be discarded
    Params:
        - image_file (PIL Image): the image the we want to analyze and evaluate
    Returns:
        - output (bool): True if the image is valid, False otherwise
    '''
    return True

def driver(vol, img, path_to_image='temp.jpg'):
    preprocess(path_to_image)
    
    pooled = block_image(path_to_image)
    pooled_img = Image.fromarray(pooled)
    pooled_img = pooled_img.resize((960, 1280))
    pooled = np.array(pooled_img)    
    
    blocks, coordinates = layout_analyze(pooled)
    entry_blocks, entry_coords = filter_blocks(blocks, coordinates)

    if entry_blocks == None:
        return False
    
    all_coords = []
    counts = []
    start_line = 1

    for entry_id, block in enumerate(entry_blocks):
        data, image_file, orig_image = gray_and_rotate(block)        
        crop_pixels = find_pixels(data, 5000)
        data = np.array(orig_image)
        count, segment_coords = data_segmentation(data, crop_pixels, str(vol) + '-' + '0' * (4 - len(str(img))) + str(img), image_file, entry_coords[entry_id], start_line) #cropping image and output
        all_coords.append(segment_coords)
        counts.append(count)
        start_line += count

    return True
        
