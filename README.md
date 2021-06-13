#
## Machine Learning with Python Project

# How green is my city ?

## Identifying Plant surface on a city with satellite images.

![greenproject](https://user-images.githubusercontent.com/72912247/121811213-02297780-cc64-11eb-9476-6ceffd05d3a8.jpeg)


**Description**
This project is about identifying green areas in a given city , the user inputs a city name and it searches for it's GPS coordinates and gives back an appropriate corresponding usable satellite image without clouds from predefined and stored dataset of satellite images, it Selects only the area corresponding to the city limits thanks to cartography data (image superposition), computes and returns the green areas in the city and the ratio of nature from the total area thanks to a machine learning algorithm.


![io](https://user-images.githubusercontent.com/72912247/121814302-897ce800-cc70-11eb-8354-e3e5d2dd85be.JPG)


**Why this project ?**

Rapid and uncontrolled Urbanization along with economic and industrial development, have increased the rate of land-cover change ,while green areas offer innovative approaches to increase the quality of urban settings, enhance local resilience and promote sustainable lifestyles, and currently the importance of urban greenspaces in an urban ecosystem is being increasingly recognized . There is a need to examine the accuracy of different algorithms for land-cover mapping in order to identify the best classifier for further applications of these areas' observation and preservation.


Challenges faced:

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
