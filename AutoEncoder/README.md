We build the autoencoder with 4 convolution layers and 4 deconvolution layers to reconstruct the image and exact the feature.
The model optimize the mse loss for the image reconstruction and the cosine loss for the similar image searching.

The order 'python autoencoder_cosine.py' to train the model while 'python autoencoder_cosine.py 1' is for testing the model.
Also, the order 'python autoencoder_cosine.py {image_path}' could find the most simlar image in the searching dataset.
