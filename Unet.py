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
    #pass


class Model(ModelDesc): #Model 
    def _get_inputs(self):
        return [InputDesc(tf.float32, (None, 1024, 1024), 'edges'),
                InputDesc(tf.int32, (None, 1024, 1024, 3), 'color')]
        #pass
                
    def _build_graph(self, input_vars):
        edges, color = input_vars # (None, 1024, 1024), (None, 1024, 1024, 3)
        self.cost = 0
        
        NF = 64
        with argscope(Conv2D, 
                      kernel_shape=4, 
                      stride=2,
                      nl=lambda x, name: LeakyReLU(BatchNorm('bn', x), name=name)):
            # encoder
            e1 = Conv2D('conv1', luminance, NF, nl=LeakyReLU)
            e2 = Conv2D('conv2', e1, NF * 2)
            e3 = Conv2D('conv3', e2, NF * 4)
            e4 = Conv2D('conv4', e3, NF * 8)
            e5 = Conv2D('conv5', e4, NF * 8)
            e6 = Conv2D('conv6', e5, NF * 8)
            e7 = Conv2D('conv7', e6, NF * 8)
            e8 = Conv2D('conv8', e7, NF * 8, nl=BNReLU)  # 1x1
        with argscope(Deconv2D, 
                      nl=BNReLU, 
                      kernel_shape=4, 
                      stride=2):
            # decoder
            e8 = Deconv2D('deconv1', e8, NF * 8)
            e8 = Dropout(e8)
            e8 = ConcatWith(e8, 3, e7)
    
            e7 = Deconv2D('deconv2', e8, NF * 8)
            e7 = Dropout(e7)
            e7 = ConcatWith(e7, 3, e6)
    
            e6 = Deconv2D('deconv3', e7, NF * 8)
            e6 = Dropout(e6)
            e6 = ConcatWith(e6, 3, e5)
    
            e5 = Deconv2D('deconv4', e6, NF * 8)
            e5 = Dropout(e5)
            e5 = ConcatWith(e5, 3, e4)
    
            e4 = Deconv2D('deconv5', e65, NF * 4)
            e4 = Dropout(e4)
            e4 = ConcatWith(e4, 3, e3)
    
            e3 = Deconv2D('deconv6', e4, NF * 2)
            e3 = Dropout(e3)
            e3 = ConcatWith(e3, 3, e2)
    
            e2 = Deconv2D('deconv7', e3, NF * 1)
            e2 = Dropout(e2)
            e2 = ConcatWith(e2, 3, e1)
    
            prediction = Deconv2D('prediction', e2, 3, nl=tf.tanh)
            
            self.cost = tf.nn.l2_loss(prediction - color, name="L2 loss")
            add_moving_summary(self.cost)
            tf.summary.image('colorized', prediction, max_outputs=10)
        
        #pass
    #pass

def get_config(): #Trainer
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # Debug
    get_data()