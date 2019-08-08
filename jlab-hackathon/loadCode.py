#!/usr/bin/env python
#
#
# ex_toy4
#
# Building on toy3 example, this adds drift distance
# information to the pixel color. It also adds a 
# random z-vertex position in addition to the phi
# angle. 
#
# The network is defined with 2 branches to calculate
# the phi and z. They share a common input layer and
# initial Dense layer then implement their own dense
# layers.
#
# Another difference from toy3 is that a final dense
# layer with a single neuron is added to each of the
# branches to calculate phi(z) parameters directly
# rather than doing that outside of the network. To
# help this, the weights feeding that last neuron are
# set to fixed weights (bin centers) and are marked
# as non-trainable.
#


import os
import sys
import gzip
import pandas as pd
import numpy as np
import math

# If running on Google Colaboratory you can uncomment the
# following and modify to use your Google Drive space.
#from google.colab import drive
#drive.mount('/content/gdrive')
#workdir = '/content/gdrive/My Drive/work/2019.03.26.trackingML/eff100_inverted'
#os.chdir( workdir )


width  = 36
height = 100

# Open labels files so we can get number of samples and pass the
# data frames to the generators later
traindf = pd.read_csv('TRAIN/track_parms.csv')
validdf = pd.read_csv('VALIDATION/track_parms.csv')
STEP_SIZE_TRAIN = len(traindf)/BS
STEP_SIZE_VALID = len(validdf)/BS

#-----------------------------------------------------
# generate_arrays_from_file
#-----------------------------------------------------
# Create generator to read in images and labels
# (used for both training and validation samples)
def generate_arrays_from_file( path, labelsdf ):

    images_path = path+'/images.raw.gz'
    print( 'generator created for: ' + images_path)

    batch_input           = []
    batch_labels_phi      = []
    batch_labels_z        = []
    idx = 0
    ibatch = 0
    while True:  # loop forever, re-reading images from same file
        with gzip.open(images_path) as f:
            while True: # loop over images in file
            
                # Read in one image
                bytes = f.read(width*height)
                if len(bytes) != (width*height): break # break into outer loop so we can re-open file
                data = np.frombuffer(bytes, dtype='B', count=width*height)
                pixels = np.reshape(data, [width, height, 1], order='F')
                pixels_norm = np.transpose(pixels.astype(np.float) / 255., axes=(1, 0, 2) )
                
                # Labels
                phi = labelsdf.phi[idx]
                z   = labelsdf.z[idx]
                idx += 1

                # Add to batch and check if it is time to yield
                batch_input.append( pixels_norm )
                batch_labels_phi.append( phi )
                batch_labels_z.append( z )
                if len(batch_input) == BS :
                    ibatch += 1
                    
                    # Since we are training multiple loss functions we must
                    # pass the labels back as a dictionary whose keys match
                    # the layer their corresponding values are being applied
                    # to.
                    labels_dict = {
                        'phi_output' :  np.array(batch_labels_phi ),
                        'z_output'   :  np.array(batch_labels_z   ),        
                    }
                    
                    yield ( np.array(batch_input), labels_dict )
                    batch_input      = []
                    batch_labels_phi = []
                    batch_labels_z   = []

            idx = 0
            f.close()


#===============================================================================
# Create training generator
train_generator = generate_arrays_from_file('TRAIN', traindf)






