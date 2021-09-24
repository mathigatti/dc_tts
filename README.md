# A TensorFlow Implementation of DC-TTS: yet another text-to-speech model

Implementation of [Efficiently Trainable Text-to-Speech System Based on Deep Convolutional Networks with Guided Attention](https://arxiv.org/abs/1710.08969). Original code taken from [here](https://github.com/Kyubyong/dc_tts).

## Requirements
  * NumPy >= 1.11.1
  * TensorFlow >= 1.3 (Note that the API of `tf.contrib.layers.layer_norm` has changed since 1.3)
  * librosa
  * tqdm
  * matplotlib
  * scipy

## Try it

Check [this](https://colab.research.google.com/drive/12qdw-PhjHatluuZQYEdfTFvOLCN9cqN-#scrollTo=VdP7_OKtrbHO) colab to train the model with a dataset of yours.
