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


import os
import zipfile

def extractFile(path,dest):
  '''function to extract zip file'''
  with zipfile.ZipFile(path, 'r') as zip:
    zip.extractall(dest)

  #checking if the folder zip exists
  assert os.path.exists(dest) == True


