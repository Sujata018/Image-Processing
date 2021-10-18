import sys
import numpy as np
from skimage import io, filters
from matplotlib import pyplot as plt


def diffuse_image(img,B,l=0.25):
    '''
    Anisotropic diffusion of an image.
    input: img - input image to be diffused
           B   - the threshold of gradient. Edges with gradient bigger than B will be preserved
           l   - weight for diffusion. Default value 0.25
    outout : oimg - diffused image with edges preserved
    '''
    
    I=np.roll(img,1,axis=0)    # shift down the image to calculate gradient in north direction = shifte image - original image
    I[0]=0
    dN=I-img
    cN=np.exp(-(dN/B)**2)      # calculate weightage of the gradient = e ^(- gradient/B) : If gradient >> B (possibly an edge), weightage is less, so less diffusion.
    
    I=np.roll(img,-1,axis=0)   # shift up the image to calculate gradient in south direction = shifte image - original image
    I[-1]=0
    dS=I-img
    cS=np.exp(-(dS/B)**2)
    
    I=np.roll(img,-1,axis=1)   # shift left the image to calculate gradient in east direction = shifte image - original image
    I[:,-1]=0
    dE=I-img
    cE=np.exp(-(dE/B)**2)

    I=np.roll(img,1,axis=1)    # shift right the image to calculate gradient in west direction = shifte image - original image
    I[:,0]=0
    dW=I-img
    cW=np.exp(-(dW/B)**2)

    oimg=img+l*(dN*cN+dS*cS+dE*cE+dW*cW) # diffuse the image as per weights decided based on gradients in 4 directions

    return oimg
    
if __name__=='__main__':

    ## Input parameters ##

    files=['Cameraman.ppm']        # Enter the names of the files to be smoothed
    C=90                           # C% weakest gradient magnitudes are noises
    l=0.25                         # weight for diffusion
    n=40                           # number of iterations for smoothing
    ## Read image(s) ##

    for file in files:
        img=io.imread(file,as_gray=True)        # read input file
        if img is None:
            sys.exit(file+': file not found.')
 
        print(file, ': size - ', img.shape)

    ## smooth images iteratively ##

        curimg=img                 # initialise input image for 1st iretation
        plt.clf()
        m=4                        # number of columns in plot showing results of diffusion
        for i in range(n):
            edge_map_sobel = filters.sobel(curimg)  # gradient image using Sobel operator
            thresh=np.percentile(edge_map_sobel,C)  # get threshold gradient as the C% least value from all gradients
            real_edges=np.where(edge_map_sobel>thresh,255,0)

            dimg=diffuse_image(curimg,thresh,l)     # diffuse the image preserving real edges

    ## plot smoothed images and histograms 
            if i%10==0:
                plt.subplot(n/10,m,m*i/10+1)                  # show image for current iteration of diffusion in the plot    
                plt.imshow(curimg,cmap='Greys_r')
                if i == 0:
                    plt.title("Image")

                plt.subplot(n/10,m,m*i/10+2)                  # show histogram of gradients and a red vertical line at C%
                plt.hist(edge_map_sobel.flatten())
                if i == 0:
                    plt.title('Histogram')
                plt.axvline(x=thresh,color='r')
            
                plt.subplot(n/10,m,m*i/10+3)                  # show edge map (gradients)
                plt.imshow(edge_map_sobel,cmap='Greys_r')
                if i == 0:
                    plt.title("Edge map")

                plt.subplot(n/10,m,m*i/10+4)                  # show edge map to be preserved (top 100-C % gradients)
                plt.imshow(real_edges,cmap='Greys_r')
                if i == 0:
                    plt.title("Edges preserved")

            io.imsave(file[:-4]+'_iter'+str(i)+'.jpg',curimg)

            curimg=dimg                             # diffused image is the input for next iteration

        plt.savefig(file[:-4]+'_AnisotropicDiffusion.png')
            
        
