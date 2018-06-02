import os
import sys
from PIL import Image
from imutils import paths
from pdb import set_trace as bp

source_path = sys.argv[1]
target_path = sys.argv[2]

for image_path in paths.list_images(source_path):

    print('Loading image', image_path)
    image = Image.open(image_path)
    filename = os.path.basename(image_path)
    base, ext = os.path.splitext(filename)

    width, height = image.size
    new_width = 512
    new_height = 512
    # Crop 5 regions from the image: 4 edges and center
    for j in range(5):

        if j==0:
            # Crop the center
            left = (width - new_width) / 2
            top = (height - new_height)/2
            right = left + new_width
            bottom = top + new_height
            print('Cropping 512x512 center from image')
        elif j==1:
            # Crop the top-left edge
            left = 0
            top = 0
            right = left + new_width
            bottom = top + new_height
            print('Cropping 512x512 top-left from image')
        elif j==2:
            # Crop the top-right edge
            left = width - new_width
            top = 0
            right = left + new_width
            bottom = top + new_height
            print('Cropping 512x512 top-right from image')
        elif j==3:
            # Crop the bottom-left edge
            left = 0
            top = height - new_height
            right = left + new_width
            bottom = top + new_height
            print('Cropping 512x512 bottom-left from image')
        elif j==4:
            # Crop the bottom-right edge
            left = width - new_width
            top = height - new_height
            right = left + new_width
            bottom = top + new_height
            print('Cropping 512x512 bottom-right from image')

        cropped_image = image.crop((left, top, right, bottom))
        cropped_image.save(target_path + '/' + base + '_' + str(j) + ext)
