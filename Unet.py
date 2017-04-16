from tensorpack import *
from tensorpack.utils.viz import *
from tensorpack.tfutils.summary import add_moving_summary

import tensorflow as tf
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
        return [InputDesc(tf.float32, (None, 1024, 1024, 1), 'edges'),
                InputDesc(tf.float32, (None, 1024, 1024, 3), 'color')]
        #pass
                
    def _build_graph(self, input_vars):
        edges, color = input_vars # (None, 1024, 1024), (None, 1024, 1024, 3)
        print edges.get_shape
        self.cost = 0
        
        NF = 64
        with argscope(BatchNorm, use_local_stat=True), \
                argscope(Dropout, is_training=True):
            # always use local stat for BN, and apply dropout even in testing
            with argscope(Conv2D, kernel_shape=4, stride=2,
                          nl=tf.identity):
                e1 = Conv2D('conv1', edges, NF)
                # e1 = Dropout(e1)
                e2 = Conv2D('conv2', e1, NF * 2)
                # e2 = Dropout(e2)
                e3 = Conv2D('conv3', e2, NF * 4)
                # e3 = Dropout(e3)
                e4 = Conv2D('conv4', e3, NF * 6)
                # e4 = Dropout(e4)
                e5 = Conv2D('conv5', e4, NF * 8)
                # e5 = Dropout(e5)
                e6 = Conv2D('conv6', e5, NF * 10)
                # e6 = Dropout(e6)
                e7 = Conv2D('conv7', e6, NF * 12)
                # e7 = Dropout(e7)
                e8 = Conv2D('conv8', e7, NF * 14)
                # e8 = Dropout(e8)
                e9 = Conv2D('conv9', e8, NF * 16, nl=BNReLU)  # 1x1
                e9 = Dropout(e9)
            with argscope(Deconv2D, nl=BNReLU, kernel_shape=4, stride=2):
                
                outputs =  (LinearWrap(e9)
                        .Deconv2D('deconv0', NF * 14)
                        .Dropout()
                        .ConcatWith(e8, 3)
                        .Deconv2D('deconv1', NF * 12)
                        .Dropout()
                        .ConcatWith(e7, 3)
                        .Deconv2D('deconv2', NF * 10)
                        .Dropout()
                        .ConcatWith(e6, 3)
                        .Deconv2D('deconv3', NF * 8)
                        .Dropout()
                        .ConcatWith(e5, 3)
                        .Deconv2D('deconv4', NF * 6)
                        # .Dropout()
                        .ConcatWith(e4, 3)
                        .Deconv2D('deconv5', NF * 4)
                        # .Dropout()
                        .ConcatWith(e3, 3)
                        .Deconv2D('deconv6', NF * 2)
                        # .Dropout()
                        .ConcatWith(e2, 3)
                        .Deconv2D('deconv7', NF * 1)
                        # .Dropout()
                        .ConcatWith(e1, 3)
                        .Deconv2D('deconv8', 3, nl=tf.tanh)())
            
            self.cost = tf.nn.l2_loss(outputs - color, name="L2_loss")
            add_moving_summary(self.cost)
            tf.summary.image('colorized', outputs, max_outputs=10)
        return outputs
        #pass
    #pass

def get_config(): #Trainer
    logger.auto_set_dir()
    dataset = get_data()
    lr = symbolic_functions.get_scalar_var('learning_rate', 2e-4, summary=True)
    return TrainConfig(
        dataflow=dataset,
        optimizer=tf.train.AdamOptimizer(lr),
        callbacks=[PeriodicTrigger(ModelSaver(), every_k_epochs=10)],
        model=Model(),
        steps_per_epoch=dataset.size(),
        max_epoch=100,
    )
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--load', help='load model')
    parser.add_argument('--view', help='view dataset', action='store_true')
    parser.add_argument('--run', help='run model on images')
    parser.add_argument('--output', help='fused output filename. default to out-fused.png')
    args = parser.parse_args()
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    

    config = get_config()
    if args.load:
        config.session_init = get_model_loader(args.load)
    SyncMultiGPUTrainer(config).train()
    
    # Debug
    #get_data()