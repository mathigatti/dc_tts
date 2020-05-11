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
import tensorflow as tf
from train import Graph
from utils import *
from data_load import load_text
from scipy.io.wavfile import write
from tqdm import tqdm
import sys

def synthesize(in_folder,out_folder):

    # Load graph
    g = Graph(mode="synthesize"); print("Graph loaded")

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

        # Load data
        while True:
            queue = set(os.listdir(in_folder))
            processed = set(map(lambda file : file.replace("wav","txt"),os.listdir(out_folder)))

            files = sorted(list(queue - processed))
            texts = []
            for file in files:
                with open(f"{in_folder}/{file}",'r') as f:
                    text = f.read()
                texts.append(text)

            if len(files) > 0:
                L = load_text(texts)
                print(L)
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
                    print("Working on file", i+1)
                    wav = spectrogram2wav(mag)
                    write(f"{out_folder}/{files[i].split('.')[0]}.wav", hp.sr, wav)

if __name__ == '__main__':
    in_folder = sys.argv[1]
    out_folder = sys.argv[2]
    synthesize(in_folder,out_folder)

