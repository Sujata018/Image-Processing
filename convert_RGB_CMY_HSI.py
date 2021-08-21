import cv2 as cv
import sys
import numpy as np
from matplotlib import pyplot as plt
from transform import colorcode as cl 

if __name__=='__main__':
    if len(sys.argv)<2:                                 # Process input file from command line argument
        file="Luna.jpg"                                 # if no input provided, process Luna.jpg
    else:
        file=sys.argv[1]
        
img=cv.imread(file)                                     # convert file to image tensor

if img is None:
    sys.exit("Could not read the image "+file)

b,g,r=img[:,:,0],img[:,:,1],img[:,:,2]                  # split into blue, green, red

# Convert RGB->CMYK 

imgK=cl.RGB2CMYK(img)                                   # convert RGB to CMYK format

# Convert CMYK->RGB and compare with original image

rgbimg=cl.CMYK2RGB(imgK)

difimg=img-rgbimg

# Convert RGB->HSI 

oimg=cl.RGB2HSI(img)                                    # convert RGB to HSI format
h,s,i=oimg[:,:,0],oimg[:,:,1],oimg[:,:,2]               # 0<=H<6.28 (2*pi), 0<=S<=1, 0<=I<=255

# Convert HSI->RGB and compare with original image

rgbimg1=cl.HSI2RGB(oimg)
difimg1=img-rgbimg1

# Display RGB and CMYK channels

plt.subplot(241)                                        # plot original image
plt.imshow(img[...,::-1])
plt.title('Original')

plt.subplot(242)                                        # plot gray scale of red image
plt.imshow([1,0,0]*img[...,::-1])
plt.title("Red")

plt.subplot(243)                                        # plot gray scale of green image
plt.imshow([0,1,0]*img[...,::-1])
plt.title("Green")

plt.subplot(244)                                        # plot gray scale of blue image
plt.imshow([0,0,1]*img[...,::-1])
plt.title("Blue")

plt.subplot(245)                                        # plot gray scale of cyan image
imgc=([0,0,1]*imgK[:,:,:3])                             # extract cyan(c) from cmyk
imgc=np.dstack([imgc[:,:,0],imgc[:,:,2],imgc[:,:,2]])   # no red and equal amount (same as c) of green and blue, to display cyan
plt.imshow(imgc)
plt.title("Cyan")

plt.subplot(246)                                        # plot gray scale of magenta image
imgm=([0,1,0]*imgK[:,:,:3])                             # extract magenta (m) from cmyk
imgm=np.dstack([imgm[:,:,1],imgm[:,:,0],imgm[:,:,1]])   # equal (m) amount of red and blue to display magenta
plt.imshow(imgm)
plt.title("Magenta")

plt.subplot(247)                                        # plot gray scale of yellow image
imgy=([1,0,0]*imgK[:,:,:3])                             # extract yellow(y) from cmyk
imgy=np.dstack([imgy[:,:,0],imgy[:,:,0],imgy[:,:,1]])   # equal (y) amount of red and green to display yellow
plt.imshow(imgy)
plt.title("Yellow")

plt.subplot(248)                                        # plot gray scale of black image
plt.imshow(imgK[:,:,3],cmap='gray')                     # extract k from cmyk, show as gray map
plt.title("Black")

plt.savefig(file[:-4]+"_Spectrum.png")

plt.show()

# Display recovered rgb image from cmyk image and the differences between original and recovered images

plt.subplot(131)                                        # plot original image
plt.imshow(img[...,::-1])
plt.title('Original')

plt.subplot(132)                                        # rgb -> cmyk -> rgb image
plt.imshow(rgbimg[...,::-1])
plt.title('RGB->CMYK->RGB')

plt.subplot(133)                                        # difference image
plt.imshow(difimg[...,::-1])
plt.title('Difference')

plt.savefig(file[:-4]+"_Diff.png")

plt.show()

# Display hsi channels

plt.subplot(131)                                        # plot gray scale of hue image
plt.imshow(h,cmap="gray")
plt.title("Hue")

plt.subplot(132)                                        # plot gray scale of saturation image
plt.imshow(s,cmap="gray")
plt.title("Saturation")

plt.subplot(133)                                        # plot gray scale of intensity image
plt.imshow(i,cmap="gray")
plt.title("Intensity")

plt.savefig(file[:-4]+"_HSI_Spectrum.png")

plt.show()

# Display recovered rgb image from hsi image and the differences between original and recovered images

plt.subplot(131)                                        # plot original image
plt.imshow(img[...,::-1])
plt.title('Original')

plt.subplot(132)                                        # rgb -> hsi -> rgb image
plt.imshow(rgbimg1[...,::-1])
plt.title('RGB->HSI->RGB')

plt.subplot(133)                                        # difference image
plt.imshow(difimg1[...,::-1])
plt.title('Difference')

plt.savefig(file[:-4]+"_Diff_rgb_hsi.png")

plt.show()

