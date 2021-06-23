# The Python Shapefile Library (PyShp) reads and writes ESRI Shapefiles in pure Python
try:
    import shapefile
except:
    from ModuleInstaller import installModule
    installModule("pyshp")
    import shapefile

'''
Adding a Polygon shape
reference tutorial: https://pypi.org/project/pyshp/#writing-shapefiles

Similarly to LineString, Polygon shapes consist of multiple polygons, and must be given as a list of polygons. The main difference is that polygons must have at least 4 points and the last point must be the same as the first. It's also okay if you forget to repeat the first point at the end; PyShp automatically checks and closes the polygons if you don't.
It's important to note that for Polygon shapefiles, your polygon coordinates must be ordered in a clockwise direction. If any of the polygons have holes, then the hole polygon coordinates must be ordered in a counterclockwise direction. The direction of your polygons determines how shapefile readers will distinguish between polygon outlines and holes.

w = shapefile.Writer('shapefiles/test/polygon')
w.field('name', 'C')

w.poly([ ... [[113,24], [112,32], [117,36], [122,37], [118,20]], # poly 1
... [[116,29],[116,26],[119,29],[119,32]], # hole 1
... [[15,2], [17,6], [22,7]] # poly 2
... ]) w.record('polygon1')

w.close()'''

def create_shapefile_from_geo_coordinates(filename, fieldname, N_lat, S_lat, W_long, E_long):
   # To create a shapefile you begin by initiating a new Writer instance, 
   # passing it the file path and name to save to:
   w = shapefile.Writer(filename)
   w.field(fieldname, 'C')

   # create a simple polygon shapefile, which is a square composed of GPS coordinates
   # Latitude and longitude of area of interest can be found with www.openstreetmap.org -> Export
   w.poly([ [[W_long, N_lat], [E_long, N_lat], [E_long, S_lat], [W_long, S_lat]]  ])
   w.record('polygon1')
   w.close()

from shapely.geometry import Point

def create_list_of_points_from_geo_coordinates(N_lat, S_lat, W_long, E_long):
   # create a simple list of polygon points corresponding to GPS coordinates of the area of interest
   # Latitude and longitude of area of interest can be found with www.openstreetmap.org -> Export
   return (Point(W_long, N_lat), Point(E_long, N_lat), Point(E_long, S_lat), Point(W_long, S_lat))


import os
import zipfile

def extractFile(path,dest):
  '''function to extract zip file'''
  with zipfile.ZipFile(path, 'r') as zip:
    zip.extractall(dest)

  #checking if the folder zip exists
  assert os.path.exists(dest) == True

import rasterio

def crop_area(src, area_shape, fileout):
  '''
  Crop a multispectral image given a shape as input. Write result in a file.
  inputs: 
    src: a rasterio dataset containing multispectral image
    area_shape : a shapefile
    fileout: output filename
  output:
    None
  '''
  out_image, out_transform = rasterio.mask.mask(src, area_shape.geometry,crop=True)
  out_meta = src.meta.copy()
  out_meta.update({"driver": "GTiff",
                 "height": out_image.shape[1],
                 "width": out_image.shape[2],
                 "transform": out_transform})
    
  with rasterio.open(fileout, "w", **out_meta) as dest:
    dest.write(out_image)


from rasterio.warp import calculate_default_transform, reproject, Resampling

def change_image_projection(filename, outfilename, dst_crs):
  '''Function comes from here:
  https://rasterio.readthedocs.io/en/latest/topics/reproject.html?highlight=warp
  Reprojecting a GeoTIFF dataset from one coordinate reference system (CRS)
  to another one'''
  with rasterio.open(filename) as src:
    transform, width, height = calculate_default_transform(
        src.crs, dst_crs, src.width, src.height, *src.bounds)
    kwargs = src.meta.copy()
    kwargs.update({
        'crs': dst_crs,
        'transform': transform,
        'nodata': 0.0, # set of 0.0 instead of -9999.0 so that green rectangle does not appear when ploting with folium
        'width': width,
        'height': height
    })

    with rasterio.open(outfilename, 'w', **kwargs) as dst:
      for i in range(1, src.count + 1):
        reproject(
                source=rasterio.band(src, i),
                destination=rasterio.band(dst, i),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=dst_crs,
                resampling=Resampling.nearest)
