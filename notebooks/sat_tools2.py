import os
import shutil
import random
import matplotlib.pyplot as plt
import ee
import osmnx as ox
import rasterio 
import numpy as np
import unidecode
import folium

from rasterio import features, mask, windows

import tensorflow as tf
from focal_loss import SparseCategoricalFocalLoss

from rasterio.plot import show
from rasterio.windows import Window

from itertools import product

def gdrive_ee_auth():
  """ Performs authentication on google drive and google earth engine if not done yet""" 

  from google.colab import drive
  drive.mount('/content/drive')

  # Earth Engine Python API

  import ee 
  # Initialize Earth Engine and athenticate to your GEE account if not done yet.
  try: 
    ee.Initialize()
  except:
    ee.Authenticate()
    ee.Initialize()

def get_area(place_name, crs_type):
    """Receives any string, queries the string using OSM Nominating API (https://nominatim.org/release-docs/latest/api/Search/) and returns an geopandas dataframe with the area geometry and area name, formated such that it can be used as unique folder name to get images downloaded from Earth Engine directly to Google drive.
    Sets the coordinate reference system as crs_type given as input.
    Returns 'None' if the query fails to find any results"""

    # search the place area, based on the given name and return bonds and exterior geometrys
    try: 
      # search the place area, based on the given name  
      place_area = ox.geocoder.geocode_to_gdf(place_name)
      print('Got the following place: ' + place_area.display_name[0])
      area_name = place_area.display_name[0].replace(' ','_').replace(',','').replace("'","")
      area_name = 'pred_'+unidecode.unidecode(area_name)[:80]
      place_area.to_crs(crs=crs_type, inplace=True)
      return area_name, place_area
    except:
      #query fails usualy due to the name to being found on OSM Database
      print('Could not find this name')
      return None, None 

def view_image(image_path):
    """Receives a full path to any image and prints 9 random areas (256x256) of the image, using rasterio"""

    def random_plot(src, my_plt, xsize= 256, ysize = 256):
        # xsize, ysize: The size in pixels of your desired window
        # Generate a random window location that doesn't go outside the image
        xmin, xmax = 0, src.width - xsize
        ymin, ymax = 0, src.height - ysize
        xoff, yoff = random.randint(xmin, xmax), random.randint(ymin, ymax)

        # Create a Window and calculate the transform from the source dataset    
        cropped = src.read(window=Window(xoff, yoff, xsize, ysize))
        show(cropped, ax=my_plt)

    print('9 random images at maximum resolution')
    dataset = rasterio.open(image_path)
    print(np.max(dataset.read()))
    print(np.min(dataset.read()))
    fig, plot_list = plt.subplots(3,3, figsize=(30,30))
    for plt1 in plot_list:
        for plt2 in plt1:
          random_plot(dataset, plt2)
    plt.show()


def tile_img(image, outpath, tile_shape, tile_dtype):
        """Create a set of tiles of the given image, on the outpath location, with the given tile_shape and tile_dtype
          Returns a list with all the tile images path
        """
    
        def get_tiles(width, height, meta):
                # get number of rows and columns (pixels) in the entire input image
                nols, nrows = meta['width'], meta['height']
                # get the grid from which tiles will be made 
                offsets = product(range(0, nols, width), range(0, nrows, height))
                # get the window of the entire input image
                big_window = Window(col_off=0, row_off=0, width=nols, height=nrows)
                # tile the big window by mini-windows per grid cell
                for col_off, row_off in offsets:
                    window = Window(col_off=col_off, row_off=row_off, width=width, height=height).intersection(big_window)
                    yield window, col_off, row_off

        #create out_dir if it does not exists and delete contentes if it existist.
        force_dirs([outpath])
        src_meta = image.meta

        #set the metadata of the tile images
        meta = src_meta.copy()
        # set the number of channels to 3 or 1, depending on if its the index image or labels image
        meta['count'] = tile_shape[2]
        # set the tile output file format to PNG (saves spatial metadata unlike JPG)
        meta['driver']='PNG'
        meta['dtype']=tile_dtype
        meta['width'] = tile_shape[0]
        meta['height'] = tile_shape[1]
        
        # tile the input image by the mini-windows
        tile_list = []
        for window, col_off, row_off in get_tiles(tile_shape[0], tile_shape[1], src_meta):
            outfile = outpath+"tile_%s_%s_.png" % (col_off, row_off)
            with rasterio.open(outfile, 'w', **meta) as outds:
                outds.write(image.read(window=window).astype(tile_dtype))
                tile_list.append(outfile)
        return tile_list

def crop_area(src_image, geometry):
  """crop any image around a geometry. Receives the full image path and a geopandas geometry"""
  with rasterio.open(src_image) as src:
    img = src.read()
    out_image, out_transform = rasterio.mask.mask(src, geometry,crop=True)
    out_meta = src.meta.copy()
  out_meta['transform'] = out_transform
  with rasterio.open(src_image, "w", **out_meta) as dest:
    dest.write(out_image)

def force_dirs(dirs, delete_existent=False):
  """Forces the creation of a list of folders given as input if they don't exist. If delete_existent is True, the existing contents of the folders are deleted"""
  for d in dirs:
        if not os.path.exists(d):
          print(d)
          os.makedirs(d)
        elif delete_existent:
          shutil.rmtree(d)
          os.makedirs(d)

def check_area(df, max_area):
    """Calculates an area in square meters of a geopandas geommetry and gives a warning if the area exceded the value of max_area given as input"""
    area = df.to_crs('ESRI:54009').area[0]
    print('total area of %0.2f km2'%(area/1e6))
    scale = max(area/max_area,1)
    if scale > 1:
        print('Maximum recommended area exceded by a factor of ' + str(scale) + '! Reduce the area of interest and try again or run cell bellow to reduce area around centroid')
        print('You can also ignore this, just be prepared to whait a long time and get large files...')
    return scale

def image_dowload(place_path, area_name, year, bounds, crs_type, force_update=False, band_sel = ('B5', 'B4', 'B2')):
  """starts a task on google earth engine that will download an composite satellite image from sentinel2 dataset
    place_path: root folder were the images will be downloaded. This path must exist in google drive, if not the image will be downloaded in the root folder of google drive.
    area_name: Unique name for the area. A folder will be created with this name in the place_path, if not existent
    year: An average of all the sat images with cloud pixel prct less that 2 withing the given year will be created.Accepted years are from 2015 to 2021
    bounds: The bounds of the image to get retrived
    band_sel: Sentinel2 Bands to download
    force_update: If False, if an image already exists on the folder, the download is skipped. 
    Returns the image path
  """
  start_date = year+'-01-01'
  end_date = year+'-12-31'

  #create the place diretory and delete existent if force_update is true
  force_dirs([place_path+area_name], delete_existent=force_update)

  img_path = place_path + area_name +'/image_'+year+'.tif'

  # set the parameters for image retrive. see here for detais: 
  # https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2#description
  image_collection_ref = "COPERNICUS/S2"
  min_cloud_cov = 2
  cloud_cover_ref = "CLOUDY_PIXEL_PERCENTAGE"
  


  #if the satellite file does not exists on google drive, perform the download from Earth Engine
  if not os.path.exists(img_path):

    # Define the filter for the given bounds, dates and max cloud covering.
    aoi = ee.Geometry.Rectangle(bounds)
    sat_scenes = ee.ImageCollection(image_collection_ref)\
        .filterBounds(aoi)\
        .filterDate(start_date, end_date)\
        .filter(ee.Filter.lt(cloud_cover_ref, min_cloud_cov))\
        .select(band_sel)

    scenes = sat_scenes.getInfo()
    img_lst = [scene['id'] for scene in scenes["features"]]

    # If there exists images landsat2 withing the given filters:
    if len(img_lst) > 0:

      #To detect the green areas we should have an average over
      #one year of images due to changes in landscape over seasons
      sat_mosaic = sat_scenes.mean().rename(band_sel) 
      ee.Projection.crs(crs_type)
      task = ee.batch.Export.image.toDrive(**{
          'image': sat_mosaic,
          'description': area_name,
          'folder': area_name,
          'fileNamePrefix': 'image_' + year, 
          'scale': 10, # resolution in meters per pixel
          'region': aoi,
          'fileFormat': 'GeoTIFF',
          'formatOptions': {
            'cloudOptimized': 'true'
          },
          'crs': crs_type
      })

      # This task will run in the background even if you close this notebook.
      # You can also check on the status of the task through the Javascript GEE interface
      # https://code.earthengine.google.com
      task.start()
      print('Earth Engine Task started: wait until the ' + year + ' satellite file is created by earthengine on google drive')
    else:
      print('No images to download! Please review Earth Engine filters ')
  else:
      print('File for year ' + year + ' already in cache')
  return img_path


def folium_preview(base_file, preview_files, show_lst=[], place_area=None):
  # Final review of the labels over detailed sat images
  # Center the map at the center of the area of interest, thanks to latitude and longitude coordinates:
  m = folium.Map(height=500)

  tile = folium.TileLayer(
          tiles = 'https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
          attr = 'Google',
          name = 'Google Satellite',
          overlay = True,
          control = True,
          opacity= 0.7
        ).add_to(m)

  _ , _ , file_path = preview_files[0]

  with rasterio.open(base_file) as base_img:
      meta = base_img.meta
      bounds = rasterio.transform.array_bounds(meta['height'], meta['width'], meta['transform'])
      #returns [west, south, east, north bounds], convert to folium format: [[N_lat, W_long], [S_lat, E_long]]
      bounds = [[bounds[3], bounds[0]], [bounds[1], bounds[2]]]
      folium.raster_layers.ImageOverlay(
          name='Original Image',
          attr='Sentinel 2 Satellite Images',
          image=np.transpose(base_img.read((1,2,3)),(1,2,0)),
          bounds=bounds,
          opacity=1,
          interactive=True,
          cross_origin=False,
      ).add_to(m)

  for year, _, img in preview_files:
    show = False
    if int(year) in show_lst:
        show = True
        print("loading prediction image for year " + year)    
        with rasterio.open(img) as pred_labels:
            folium.raster_layers.ImageOverlay(
                name='Predicted labels for year ' + year,
                image=pred_labels.read(1),
                bounds=bounds,
                opacity=0.4,
                interactive=True,
                show=show,
                cross_origin=False,
                colormap=lambda x: (0, 1, 0, x),
            ).add_to(m)
  
  if (place_area is not None):
    folium.GeoJson(place_area, style_function=lambda x: {'fillColor': '#00000000'}, name="Area contour").add_to(m)
 
  m.add_child(folium.LayerControl())
 
  m.fit_bounds(bounds)

  return m

def get_dataset(filenames, height, width, threads=5):
  """Receives a list of png tiles paths, their target height and width and returns a dataset with tupes of image data and path"""

  def process_img_path(fname):
    """receives a png image path and returns a tensor with the image and the respective image path"""  
    #gets the raw data of the image file
    img_str = tf.io.read_file(fname)
    #decode as a png image, of dtupe uint16
    img = tf.image.decode_png(img_str, dtype= tf.uint16)
    #ensure that the image is size (height, width), filling with 0 pads if smaller
#    img = tf.image.pad_to_bounding_box(img, 0, 0, height, width)
    #perform image standartization
    img = tf.image.per_image_standardization(img)
    return img, fname
         
  # Create a dataset from the filenames 
  dataset = tf.data.Dataset.from_tensor_slices(filenames)
  # Map our preprocessing function to every element in our dataset, taking
  # advantage of multithreading
  dataset = dataset.map(process_img_path, num_parallel_calls=threads)  
  return dataset

def predict_with_tile(input_file, output_file, model, temp_dir = 'tmp_tiles/', tile_shape= [224,224,3], dtype='uint16', force_update=False):
  """ input_file: Full path of the tif image to use on the prediction, any size
      out_file: Full path to save the tif image with the Pixelwise  classification
      model: Any treined tensorflow model to the pixelwise classificationprediction 
      temp_dir: A temporary path to save the tiles: This directory will be created and deleted
      tile_shape: The shape of the tiles to divide the original image, according the input shape of the model
      dtype: The dtype of the tiles, according to the dtype used in the model training.
      force_update: If False and the output_file alredy exists no changes are made to it. 
  """
 
  if ((not os.path.exists(output_file)) or force_update):
    print('Performing the prediction')
    # if the output_file does not exists or if force_update is True, do the prediction
    with rasterio.open(input_file) as src_img:
      #Open the input image, and tile in small sub-images using tile_img function and based on tile_shape and dtype inputs
      src_meta = src_img.meta.copy()
      x_filenames = tile_img(src_img, temp_dir, tile_shape, dtype)
    
    if len(x_filenames) > 0:
      #if we have any tiled image get a dataset with all the images

      height, width = tile_shape[0], tile_shape[1]
      temp_ds = get_dataset(x_filenames, height, width)
      
      #set the prediction image size to be a full multiple of the tile size, and initialize to zeros.
      full_height = (int(src_meta['height']/height) +1)*height
      full_width = (int(src_meta['width']/width)+1)*width
      full_pred = np.zeros((full_height, full_width,  1))

      for img, file in iter(temp_ds):
            #gets the file name of the tile
            fname = file.numpy().decode('UTF-8')
            #returns the location of the tile in the main image (nrow, ncol) based on name
            loc = [int(i) for i in os.path.basename(fname).split('_')[1:3]]
            nrow_s, nrow_e = loc[1], min(loc[1]+height, full_height)
            ncol_s, ncol_e = loc[0], min(loc[0]+width, full_width)                                                                             

            img = img[tf.newaxis, ...] #add a new axis for batch 
            pred_mask = model.predict(img) #perform the prediction
            pred_mask = tf.argmax(pred_mask, axis=-1) #Returns the index with the largest value
            pred_mask = pred_mask[..., tf.newaxis] #Adds a new axis (channel)
            pred_mask = tf.keras.backend.eval(pred_mask) #Get the image value from the tensor
            full_pred[nrow_s:nrow_e,ncol_s:ncol_e] = pred_mask #Burn the predicted tile on the large image 
      
      full_pred = full_pred[:src_meta['height'],:src_meta['width'],:] #Remove the extra areas from the image

      #save the predicted image with the same metadata as the original image
      with rasterio.open( output_file, 'w', 
                          driver='Gtiff',
                          width=src_meta['width'], height=src_meta['height'],
                          count=1,
                          crs=src_meta['crs'],
                          transform=src_meta['transform'],
                          dtype='uint8') as pred:
          pred.write(full_pred.transpose((2,0,1)))
  else:
    print("Prediction already exists, skipping...")
    with rasterio.open(output_file) as pred:
          full_pred = pred.read(1)

  pred_green = np.sum(full_pred)/(np.size(full_pred))
  return pred_green

def norm(arr):
        """Recives a numpy array and return normalized value scaled and as type utin16"""
        max = np.max(arr)
        min = np.min(arr)
        arr = (arr - min)/(max-min)
        arr= (65535*arr).astype('uint16')
        return arr

def bandstack(b1, b2, b3):
    """receives any 3 arrays (bands) and returns a normalized and stacked array of the 3 bands"""  
    b1 = b1.reshape((b1.shape[0], b1.shape[1], 1))
    b2 = b2.reshape((b2.shape[0], b2.shape[1], 1))
    b3 = b3.reshape((b3.shape[0], b3.shape[1], 1))
    stack = np.dstack((b1, b2, b3))
    stack=norm(stack)
    return stack

def save_stack(stack, stack_file, src_prf, channels, force_overwrite=False):
    """Receives an array image (stack), a file path, 
    a reference to the original image metadata, and the number of channels to save 
    If force_overwrite is True if file does not exists"""

    stack_t = stack.transpose(2,0,1)
    if ((not os.path.exists(stack_file)) or force_overwrite):
      stack_out=rasterio.open(stack_file, 'w', driver='Gtiff',
                                width=src_prf.width, height=src.height,
                                count=channels,
                                crs=src_prf.crs,
                                transform=src_prf.transform,
                                dtype='uint16')
      stack_out.write(stack_t)
      stack_out.close()

def save_label(geo, src_prf, file_path, class_dict):
    """Receives a geopandas dataset, a reference image metadata, a file path to save and a class dictionary
    And burns the dataset on a tiff file according to the corresponding name vale on the dataset"""
    src_prf.update(dtype=rasterio.uint8, count=1, driver='Gtiff')

    #convert the class identifier column to type integer
    geo['id']  = geo.name.map(class_dict)
    # pair the geometries and their integer class values
    shapes = ((geom,value) for geom, value in zip(geo.geometry, geo.id))
  
    labels = features.rasterize(shapes=shapes, out_shape=(src_prf['height'], src_prf['width']), transform=src_prf['transform'], fill=0, all_touched=True, dtype=rasterio.uint8)

    print("Values in labeled image: ", np.unique(labels))
    with rasterio.open(file_path, 'w', **src_prf) as labels_out:
        labels_out.write(labels.astype(rasterio.uint8), 1) # write band 1 in tif file