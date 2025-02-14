from matplotlib import pyplot as plt
import numpy as np

# Make sure image dimensions are even numbers
def even(img):
    if (img.shape[0]%2 != 0):
        img = img[0:img.shape[0]-1,0:img.shape[1]]
    if (img.shape[1]%2 != 0):
        img = img[0:img.shape[0],0:img.shape[1]-1]
    return img


# Blur image
def average_pixels(img, grayscale = True):

    average = np.average(img, axis = (0,1))
    height = img.shape[0]
    width = img.shape[1]

    if (grayscale):
        averaged_image = np.zeros((height, width), np.uint8)
        averaged_image[:] = (average)
    else:
        averaged_image = np.zeros((height, width, 3), np.uint8)
        averaged_image[:] = (average[0], average[1], average[2])

    return averaged_image

# Split grayscale image in two
def split_image_grayscale(img):

    parts = []

    # split horizontally or vertycally, depending which is longer
    if (img.shape[0] > img.shape[1]):
        for i in range(0,img.shape[0],int(img.shape[0]/2)):
            parts.append(img[i:i+int(img.shape[0]/2), :])
    else:
        for i in range(0,img.shape[1],int(img.shape[1]/2)):
            parts.append(img[:, i:i+int(img.shape[1]/2)])
    return parts

# Cut image into parts depending on number of inputs
def deconstruct_image(img, inputs):
    parts = [img]
    new_parts = []

    # For each input
    for input in range(inputs-1):
        # If all original parts split, continue splitting already splitted parts
        if (not parts):
            parts = new_parts
            new_parts = []
        
        # Ensure all dimensions are even to allow splitting
        parts = list(map(even, parts))
        
        # Split image and append parts to new_parts array
        for splits in split_image_grayscale(parts[0]):
            new_parts.append(splits)
        
        # Remove already split part
        parts = parts[1:]

    # parts and new parts need to be merged, depending on number of splits performed
    parts = new_parts + parts
    
    return parts

# Reconstruct image from parts
def reconstruct_image(parts):
    i = 0
    while (len(parts)>1):

        for p in parts:
            print(p.shape)
        print("------")

        if (parts[i].shape[0] <= parts[i].shape[1]):
            img = np.concatenate((parts[i], parts[i+1]), axis=0)
        else:
            img = np.concatenate((parts[i], parts[i+1]), axis=1)

        # Cut out processed elements and insert merged version
        parts = parts [:i] + parts [i+2:]
        parts.insert(i,img)

        # If odd number of neurons, do not process last element
        if (i >= len(parts)-2):
            i = 0

        # If not all elements are split equally, start from beginning when end of equal shapes is reached
        elif (parts[i].shape == parts[i + 1].shape):
            i = 0

        # Go to next element
        else:
            i += 1

    return parts[0]