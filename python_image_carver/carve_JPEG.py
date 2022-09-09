"""
This script carves out contiguous JPEG files from a .raw file used in the 2006 digital forensics competition 
"""

import re
from PIL import Image
from io import BytesIO

#Encode image start and end bytes
img_start = b'\xFF\xD8\xFF\xE0'
img_end = b'\xFF\xD9' 

#Open the file as a byte array
raw_bytes = open('dfrws-2006-challenge.raw', 'rb').read()

#Get a list of all potential image start tags
soi_loc = []
for m in re.finditer(re.escape(img_start), raw_bytes):
    soi_loc.append(m.start())

#Get a list of all potential image end tags
eoi_loc = []
for m in re.finditer(re.escape(img_end), raw_bytes):
    eoi_loc.append(m.start())

#Loop through tag lists and carve our contiguous potential JPEG strings
i = 1
for soi in soi_loc:
    for eoi in eoi_loc:
       if eoi > soi:
        poss_img = raw_bytes[soi:eoi]
        try:
            img = Image.open(BytesIO(poss_img))
            img.verify()
            carve_obj = open("carved_img_" + str(i) + ".jpg",'wb').write(poss_img)
            i += 1
            print ("Found a valid JPEG file starting at offset %s" %(str(soi)))
            break
        except:
            break