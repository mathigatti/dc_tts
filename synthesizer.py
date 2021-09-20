# -*- coding: utf-8 -*-
# /usr/bin/python2
'''
By kyubyong park. kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/dc_tts
'''

from __future__ import print_function

import os

from hyperparams import Hyperparams as hp
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
from train import Graph
from utils import *
from data_load import load_text
from scipy.io.wavfile import write
from tqdm import tqdm
import sys

import warnings
warnings.filterwarnings('ignore')

# Load graph
g = Graph(mode="synthesize"); print("Graph loaded")

def synthesize(texts,out_folder):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Restore parameters
        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Text2Mel')
        saver1 = tf.train.Saver(var_list=var_list)
        saver1.restore(sess, tf.train.latest_checkpoint(hp.logdir + "-1"))
        print("Text2Mel Restored!")

        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'SSRN') + \
                   tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'gs')
        saver2 = tf.train.Saver(var_list=var_list)
        saver2.restore(sess, tf.train.latest_checkpoint(hp.logdir + "-2"))
        print("SSRN Restored!")

        if len(texts) > 0:
            L = load_text(texts)

            max_T = sum([len(text) for text in texts])*2
            # Feed Forward
            ## mel
            Y = np.zeros((len(L), hp.max_T, hp.n_mels), np.float32)
            prev_max_attentions = np.zeros((len(L),), np.int32)
            for j in tqdm(range(max_T)):
                _gs, _Y, _max_attentions, _alignments = \
                    sess.run([g.global_step, g.Y, g.max_attentions, g.alignments],
                              {g.L: L,
                              g.mels: Y,
                              g.prev_max_attentions: prev_max_attentions})
                Y[:, j, :] = _Y[:, j, :]
                prev_max_attentions = _max_attentions[:, j]

            # Get magnitude
            Z = sess.run(g.Z, {g.Y: Y})

            for i, mag in enumerate(Z):
                wav = spectrogram2wav(mag)
                write(f"{out_folder}/{i}.wav", hp.sr, wav)

if __name__ == '__main__':
  texts = [sys.argv[1]]
  out_folder = sys.argv[2]
  synthesize(in_folder,out_folder)
