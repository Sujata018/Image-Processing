# Image-Processing

## P2-code --> implementation of paper 05746646 "Partitioning Histopathological Images: An Integrated Framework for Supervised Color-Texture Segmentation and Cell Splitting"  proposed by Hui Kong*, Metin Gurcan, Senior Member, IEEE, and Kamel Belkacem-Boussaid, Senior Member, IEEE

Segmentation results from this project:

<table>
  <tr>
    <td>Original image</td>
    <td>Segmented image</td>
  </tr>
  <tr>
    <td><img src="https://github.com/Sujata018/Image-Processing/blob/main/images/P2/national-cancer-institute.jpg" width=400 height=400></td>
    <td><img src="https://github.com/Sujata018/Image-Processing/blob/main/images/P2/national-cancer-institute_segmented.bmp" width=400 height=400></td>
  </tr>
 </table>

![Original image](https://github.com/Sujata018/Image-Processing/blob/main/images/P2/national-cancer-institute.jpg) ![Segmented image](https://github.com/Sujata018/Image-Processing/blob/main/images/P2/national-cancer-institute_segmented.bmp)

convert_pgm_binary_to_ASCII.py --> Convert a binary pgm file to ASCII format

convert_pgm_ASCII_to_binary.py --> Convert an ASCII file to binary

histogramEqualization.py --> Increase contrast of an image using histogram equalization

image_compression_binary.py --> Compress binary image using bit slicing

image_compression_ASCII.py --> Compress ASCII image using bit slicing

imageSegmentation.py --> Segment an image (in binary format) using thresholding (Ostu's method and basic global thresholding method)

imageSegmentation_ascii.py --> Segment an image (in ASCII format) using thresholding (Ostu's method and basic global thresholding method)

pgm_binary_to_matrix.py --> read a P5 format Portable Gray Map file into a matrix
