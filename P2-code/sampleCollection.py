import sys
import cv2 as cv
import numpy as np


'''
Create training image patches - positive(cellular) and negative(extra cellular region) of 11x11 local neighbourhood 
'''
# Samples from 43601_HE.bmp
#marksPositive=[[665,454],[685,891],[1246,657],[1596,750],[1867,544],[1690,1209],[553,897],[756,1406],[673,1458],[619,1469],[1831,1268],[1695,1202],[1860,927],[2246,1616],[2088,1702],[1489,1561],[1516,1695],[1294,678],[368,1678],[1704,1200]]
#marksNegative=[[752,940],[28,44],[2156,310],[1562,1606],[36,1356],[670,1032],[752,1240],[1122,1296],[1224,1288],[1340,1180],[1962,1208],[2096,1122],[1786,1050],[1544,1574],[2178,1600],[1304,1670],[352,1640],[220,770],[2160,364],[2234,44]]

# Samples from national-cancer-institute-lsxKuARrQXI-unsplash_HE.bmp
#marksPositive=[[192,684],[200,702],[330,722],[540,668],[1246,734],[1342,694],[1984,690],[2380,712],[2612,920],[2428,1044],[2754,1260],[2952,1184],[278,1218],[372,354],[950,24],[1120,26],[1696,18],[2220,16],[2698,10],[2848,86],[1242,744],[2024,660],[2608,2748],[2638,2298],[2396,2180],[1662,2618],[198,2784],[482,2782],[2434,1042],[2122,850]]
#marksNegative=[[2144,758],[2518,502],[1400,128],[1574,270],[1940,812],[552,2350],[40,1332],[42,1620],[54,2062],[42,2364],[60,2442],[188,2420],[1060,2940],[996,2708],[1814,2732],[2932,2032],[982,2592],[1934,1450],[1526,1370],[848,1558],[786,1440],[656,1454],[264,1778],[50,1808],[392,2244],[218,2516],[2898,1508],[812,2382],[1064,2574],[1778,2564]]

# Samples from lung1.jpg
#marksPositive=[[],[],[],[],[],[],[],[],[],[]]
#marksNegative=[[],[],[],[],[],[],[],[],[],[]]

# Samples from lung4.jpg
#marksPositive=[[72,52],[188,303],[372,156],[431,140],[564,355],[666,454],[719,615],[343,727],[76,701],[19,554]]
#marksNegative=[[728,81],[41,234],[109,300],[367,311],[160,599],[244,683],[369,644],[459,615],[514,643],[593,737]]

# Samples from lung3.jpg
#marksPositive=[[337,20],[219,66],[457,160],[434,241],[175,371],[346,446],[616,414],[624,457],[508,621],[199,653]]
#marksNegative=[[92,588],[65,658],[560,565],[637,532],[703,550],[571,558],[677,647],[690,445],[638,242],[707,98]]

# Samples from lung5.jpg
marksPositive=[[134,66],[151,193],[247,209],[183,409],[335,215],[470,186],[563,261],[528,449],[627,500],[355,656]]
marksNegative=[[34,321],[327,395],[381,498],[389,571],[546,668],[718,675],[517,655],[531,215],[699,135],[574,11]]

file='lung5.jpg'
n_start=71
n_samples=10

img=cv.imread(file)                    # convert histogram equalized file to image tensor

if img is None:
    sys.exit("Could not read the image "+file)

print(img.shape)
print(len(marksPositive))
print(len(marksNegative))

for i in range(n_start,n_start+n_samples):
    x=marksPositive[i-n_start][1]
    y=marksPositive[i-n_start][0]
    imgcropPos=img[x:x+11,y:y+11]
    x=marksNegative[i-n_start][1]

    y=marksNegative[i-n_start][0]
    imgcropNeg=img[x:x+11,y:y+11]

    cv.imwrite('PositiveSample'+str(i)+'.bmp',imgcropPos) # Create positive training sample image
    cv.imwrite('NegativeSample'+str(i)+'.bmp',imgcropNeg) # Create negative training sample image
