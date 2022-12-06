"""
Code to remove watermarks from images by overriding the pixel values.
Takes image background colour from a selected pixel and uses this value to overwrite pixels in a rectangular subsection of the image.
"""

from PIL import Image
import glob
import os

source_base = "./Images/8"
dest_base = "./Images"

# Iterate over folders
for i in range(1, 9):
    dest = dest_base + str(i)
    source = source_base + str(i)
    print(i)
    for img in glob.glob(os.path.join(source, "*.png")):
        name = os.path.basename(img)
        
        # Removing watermark from image
        
        im = Image.open(source+"/"+name)
        
        im = im.convert("RGB")
        pix = im.load()
        
        new_color = pix[0, 0]
        for x in range(0, 700):
        	for y in range(1030, 1080):
        		im.putpixel((x, y), new_color)
        
        		
        # Image file name operations	
        print(name)	
        name = name.split("1920")[1]
        name = str(name)+'.png'
        for i in range(len(name)):
        	if(name[i].isalpha()):
        		name = name[:i]+name[0].upper()+name[1:]
        		break
        print(name)
        im.save(dest+"/"+name)

# Code for single images      
"""
from PIL import Image
im = Image.open("./Pokewalls/8/oras1920x1080.jpg")
im = im.convert("RGB")

# If performing image-level operations
# width, height = im.size

pix = im.load()

# Get the backgorund colour from the top-left corner of the image
# Alternatively select an RGB value such as 0, 0, 0
new_color = pix[0, 0]

# Wipe the selected pixel range with the background colour as replacement
for x in range(0, 1920):
    for y in range(1040, 1080):
        im.putpixel((x, y), new_color)
im.save("./New_Pokewalls/8/ORAS.png")

"""

