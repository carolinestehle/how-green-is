#
## ML with Python Project

# How green is my city ?

![greenproject](https://user-images.githubusercontent.com/72912247/121811213-02297780-cc64-11eb-9476-6ceffd05d3a8.jpeg)


## Introduction
**Project summary **

Why this project ?

Rapid and uncontrolled Urbanization along with economic and industrial development, have increased the rate of land-cover change ,while green areas offer innovative approaches to increase the quality of urban settings, enhance local resilience and promote sustainable lifestyles, and currently the importance of urban greenspaces in an urban ecosystem is being increasingly recognized . There is a need to examine the accuracy of different algorithms for land-cover mapping in order to identify the best classifier for further applications of these areas' observation and preservation.


## Identifying Plant surface on city with satellite images.

**Project summary:**

Create an application to give recent plant surface of a given city, and ratio of nature/ buildings using satellite images.

**What is already existing ?**

State of the art

The subject of land cover has been widely covered in the litterature. There are many techniques using machine learning to make land classification, detect trees or human constructions like buildings. (see algorithmic part).

Existing similar application ?

- Some companies propose products to do similar research&#39;s, but these products are not free (examples :
  - [Forests and Sustainable Land Use - Satellite Imagery and Monitoring | Planet](https://www.planet.com/markets/forestry/)
  - [Data Science &amp; Spatial Analysis | Use Spatial Science to Transform Data into Action (esri.com)](https://www.esri.com/en-us/arcgis/products/spatial-analytics-data-science/overview))
- There is a French site, created by society Kermap , https://www.nosvillesvertes.fr/ with a free application which gives the ratio of nature/buildings for each city in France but the last 3 years are missing. The application seems not to make dynamic computation, the data seems to have been computed in advance and the application just prints the stored results.

**What do we propose:**

To implement a free application :

- Given a city name as input

Which in a reasonable time (this must be quantified) :

- Searches for GPS coordinates and appropriate corresponding recent and usable satellite image (less than 1 year, without clouds) from predefined and stored dataset of satellite images as a first step
- Selects only the area corresponding to the city limits thanks to cartography data (image superposition)
- Is able to compute and return the plant surface and the ratio of nature/ buildings of the city thanks to a machine learning algorithm
- Plots and returns the plant surface of the city in a colored image

Areas of possible improvements / additional features:

- Automatic selection of a recent and usable satellite image without clouds :
  - cloud/ no clouds classification
  - Download dynamically the images from web databases
- Changes in time:
  - Computation of the evolution of the ratio nature/ buildings
  - Highlight of the changes in the output image
- Finer classification inside satellite image
- User&#39;s interface

Our Challenges:

Need to learn a lot of things about satellite imagery:

- type of data formats
- satellite missions (locations covered, frequency of clichés, type of image, spectral characteristics, resolution of images...)

Explore the tools available and learn how to use them (ex: CEREMA tools are now online and free since 2020, Earthpy...)

Identify the appropriate dataset(s) for training

Implement, test and compare various classification algorithms on plant surface, and potentially develop a mixed implementation.

Cross check our results with past data (ex: Corine land cover dataset, geoportail...)

Search how to make the computation efficient (cloud computing, parallelization...)

**Datasets**

There are plenty of datasets :

Free labeled datasets with satellite images (to be used for training)

- Kaggle satellite image datasets with labeled data
- Spacenet (17355 images, 5 areas of interest, which are big cities around the world: Rio de Janeiro, Vega, Paris, Shangai, Khartoum)

Free Unlabeled datasets with satellite images (to be used for inference)

- Copernicus satellite images (Sentinel1 and 2) : 1 image every 5 or 6 days for 1 given earth location
-

Cartography/map datasets:

- Open Street Map
- IGN
- Corine land cover

Datasets opened for education and research purposes:

- Planet
- Radiant Earth Foundation

**libraries to be used :**

- Numpy.
- Pandas, geopandas
- Matplotlib
- Scikit-learn, useful for general numeric data types.
- scikit-image
- Keras
- TensorFlow.

**Algorithms could be used:**

- Convolutional neural network (ConvNets or CNNs) : one of the main categories to do images recognition, images classifications. Objects detections, recognition faces etc.
- Mask R-CNN tools allows to train the model and use it for a detection of objects, It can be seen as a supervised classification using convolutional neural networks.
- The Satellite Imagery Multiscale Rapid Detection with Windowed Networks (SIMRDWN) codebase combines some of the leading object detection algorithms into a unified framework designed to detect objects both large and small in overhead imagery
- U-Net, a kind of Convolutional Neural Networks (CNN) approach.
