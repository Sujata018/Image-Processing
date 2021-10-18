# Image-Processing

### P2-code  
implementation of paper 05746646 "Partitioning Histopathological Images: An Integrated Framework for Supervised Color-Texture Segmentation and Cell Splitting"  proposed by Hui Kong*, Metin Gurcan, Senior Member, IEEE, and Kamel Belkacem-Boussaid, Senior Member, IEEE

Segmentation results from this project:

<table>
  <tr>
    <td>Original image</td>
    <td>Segmented image</td>
  </tr>
  <tr>
    <td><img src="https://github.com/Sujata018/Image-Processing/blob/main/images/P2/national-cancer-institute.jpg" width=400 height=300></td>
    <td><img src="https://github.com/Sujata018/Image-Processing/blob/main/images/P2/national-cancer-institute_segmented.bmp" width=400 height=300></td>
  </tr>
  <tr>
    <td><img src="https://github.com/Sujata018/Image-Processing/blob/main/images/P2/43601.jpg" width=400 height=300></td>
    <td><img src="https://github.com/Sujata018/Image-Processing/blob/main/images/P2/43601_HE_segmented.bmp" width=400 height=300></td>
  </tr>
 </table>
 
 (Please check [Wiki] https://github.com/Sujata018/Image-Processing/wiki or P2-implementation.pptx for more details.)
 
--------
### Package : transform

<table>
  <tr>
    <td>Main program</td>
    <td>Module used from package</td>
    <td>Functionality</td>
  </tr>
  <tr>
    <td>convert_RGB_CMY_HSI.py</td>
    <td>colorcode.py</td>
    <td>Transforms input image from command line from RGB to CMYK, then transforms back to RGB, then transfrms to HSI, then transforms back to RGB. Shows differences between original and recovered images after conversion.</td>
  </tr>
 </table>
 
 ### Package : enhancement
 
 <table>
  <tr>
    <th>Source</th>
    <th>Description</th>
  </tr>
  <tr>
    <td>adaptiveEdgeSmoothingOpenCV.py</td>
    <td>Pakgage used : OpenCV <br> Implementation of edge adaptive smoothing</td>
  </tr>
  <tr>
    <td>anisotropicDiffusion_skimage.py</td>
    <td>Pakgage used : skimage <br> Implementation of anisotropic diffusion</td>
  </tr>
</table>


### Other codes

convert_pgm_binary_to_ASCII.py --> Convert a binary pgm file to ASCII format

convert_pgm_ASCII_to_binary.py --> Convert an ASCII file to binary

histogramEqualization.py --> Increase contrast of an image using histogram equalization

image_compression_binary.py --> Compress binary image using bit slicing

image_compression_ASCII.py --> Compress ASCII image using bit slicing

imageSegmentation.py --> Segment an image (in binary format) using thresholding (Ostu's method and basic global thresholding method)

imageSegmentation_ascii.py --> Segment an image (in ASCII format) using thresholding (Ostu's method and basic global thresholding method)

pgm_binary_to_matrix.py --> read a P5 format Portable Gray Map file into a matrix
