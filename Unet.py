from tensorpack import *
import glob, os, natsort
import matplotlib.pyplot as plt
import cv2
import skimage.io
import skimage.segmentation
import argparse

BATCH_SIZE = 1
QUEUE_SIZE = 4

def get_data(): # Dataflow
    # Query the list of images
    folder = 'data/Voronoi/train/'
    #for dirName, subdirList, fileList in os.walk(folder): pass
    #fileList = natsort.natsorted(fileList) # Sort naturally
    imgs = glob.glob(os.path.join(folder, '*.png'))
    print imgs
    
    # Read image to dataset ds    
    ds = ImageFromFile(imgs, channel=3, shuffle=True)
    #membrs = skimage.segmentation.find_boundaries(labels, mode='inner')
    ds = MapData(ds, lambda dp: [skimage.segmentation.find_boundaries(dp[0], mode='inner')[:,:,0], dp[0]])
    ds = BatchData(ds, BATCH_SIZE)
    ds = PrefetchData(ds, QUEUE_SIZE)
    ds = PrintData(ds, num=2) # only for debugging
    return ds
    pass


class Unet(ModelDesc): #Model 
    def _get_inputs(self):
        pass
    def _build_graph(self, input_vars):
        pass
    pass

def get_config(): #Trainer
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # Debug
    get_data()