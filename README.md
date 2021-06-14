#
## Machine Learning with Python Project

# How green is my city ?

## Identifying Plant surface on a city with satellite images.

![greenproject](https://user-images.githubusercontent.com/72912247/121811213-02297780-cc64-11eb-9476-6ceffd05d3a8.jpeg)


**Description**
This project is about identifying green areas in a given city , the user inputs a city name and it searches for it's GPS coordinates and gives back an appropriate corresponding usable satellite image without clouds from predefined and stored dataset of satellite images, it Selects only the area corresponding to the city limits thanks to cartography data (image superposition), computes and returns the green areas in the city and the ratio of nature from the total area thanks to a machine learning algorithm.


**Why this project ?**

Rapid and uncontrolled Urbanization along with economic and industrial development, have increased the rate of land-cover change ,while green areas offer innovative approaches to increase the quality of urban settings, enhance local resilience and promote sustainable lifestyles, and currently the importance of urban greenspaces in an urban ecosystem is being increasingly recognized . There is a need to examine the accuracy of different algorithms for land-cover mapping in order to identify the best classifier for further applications of these areas' observation and preservation.


**Project definition - Flow chart**
![flow](https://user-images.githubusercontent.com/72912247/121818590-65c59c00-cc88-11eb-8c6d-80f02a92b048.JPG)


**Datasets**

There are plenty of datasets :

Free Unlabeled datasets with satellite images used :

- Copernicus satellite images (Sentinel1 and 2) : 1 image every 5 or 6 days for 1 given earth location
- Landsat (US) :moderate spatial-resolution (30-meter) imagery that provides large areas of repeated data coverage at a scale that enables users to see detailed human-scale processes, such as urbanization, but not individual houses.

Cartography/map datasets:

- Open Street Map
- IGN
- Corine land cover


**Project Steps:**

![steps](https://user-images.githubusercontent.com/72912247/121821154-79c4ca00-cc97-11eb-87d9-c45aab7ced9a.JPG)



**Step 1**

![step1](https://user-images.githubusercontent.com/72912247/121911112-bd6b1280-cd2f-11eb-97d7-c30af0f82da2.JPG)


**Step 2**

**Using OSM to Create sat images and OSD shapes**

1.Receive any coordinate set or place name and get a coordinate set for data download.

2.Change the area of interest interactively with folium !!NEW!!

3.Retrieve a geopadas dataframe with a set of polygons or multipolygons of green areas identified in the Open Street Map public database.

4.Retrieve images from earth engine of of Sentinel 2 Satellites/Landsat8 

5.Create a mask of 0 and 1 for green areas of the coordinate area identified above and of the dataframe created on 2 or any other geopandas dataframe with a list of polygons identifing green areas and save as a one band TIF file on google drive.


![io](https://user-images.githubusercontent.com/72912247/121821733-2a809880-cc9b-11eb-8fe2-20c48ef24d6a.JPG)



**Testing different ML and DL algorithms:**

**implementing UNet:**
a convolutional network model classically used for biomedical image segmentation with the Functional API
![unet](https://user-images.githubusercontent.com/72912247/121819354-da9ad500-cc8c-11eb-9bb8-3737330143e9.png)

The network consists of a contracting path and an expansive path, which gives it the u-shaped architecture. The contracting path is a typical convolutional network that consists of repeated application of convolutions, each followed by a rectified linear unit (ReLU) and a max pooling operation. During the contraction, the spatial information is reduced while feature information is increased. The expansive pathway combines the feature and spatial information through a sequence of up-convolutions and concatenations with high-resolution features from the contracting path 

*Training
*Inferance insert results (accuracy/loss/confusin matrix) / input image-predicted image .........


**Pixel-Wise Classification Using Deep Neural Networks:**
A basic architecture of the NN 
![bands](https://user-images.githubusercontent.com/72912247/121820860-4a14c280-cc95-11eb-8648-6ab28852ba7d.jpeg)

Pixel-wise classification is a fundamental task in remote sensing that aims at assigning a semantic
class, e.g., vegetation, accurately to every individual pixel of an image.



*Training
*Inferance insert results (accuracy/loss/confusin matrix) / input image-predicted image .........

**Step 3/ under-construction**

**Conclusion - Next steps:**



**Demo-Link:**
