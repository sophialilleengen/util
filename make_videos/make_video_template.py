## install imageio with eg pip install imageio

import imageio # documentation: https://imageio.readthedocs.io/en/stable/

import glob
import re

###### these need to be adapted
# directory where your images are saved in as numerically indexed .png files
dirname = './' 
# frames per second
fps = 5 
# name of the files you are saving (gif will have .gif ending, videp a .mp4 ending)
savename = './video'

######

### this function sorts your input files numerically (0,1,,...9,10,11...99, 100,...)
def sorted_nicely( l ):
    """Sorts the given iterable in the way that is expected.
        Required arguments:
        l -- The iterable to be sorted
    """

    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key = alphanum_key)
    
### prepare file list and load images

# read in files
files = [f for f in glob.glob(dirname + '**.png', recursive=True)] # images saved as png
files = sorted_nicely(files)

images = []

for filename in files:
    images.append(imageio.imread(filename)) # load images with imageio
    
### make video / gif
# documentation for imageio.mimwrite(): https://imageio.readthedocs.io/en/stable/userapi.html?highlight=mimwrite#imageio.mimwrite

# make gif
imageio.mimwrite(savename + '.gif', images ,format='GIF-FI', fps = fps)

# make video
imageio.mimwrite(savename + '.mp4', images , fps = fps)
