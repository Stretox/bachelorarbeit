# Implementation of a hair-flow predicting vision-transformer

The goal of this implementation is to predict a hair-point-movement direction from normalized-flow-maps by associating current-points with previous points and thus learning a global hair motion.

In the tests the motion was implemented as a classification tasks between the query points and the previous points, by associating embeddings between them utilizing a cross-attention mechanism.

# Information about this Implementation

In the current state the Transformer is unable to reliably predict previous point connections. This might be due to several factors like inaffective or wrongly implemented losses, bad input data, unreliable embeddings or something entirely else.

# ToDos

- [x] Refactor Code to improve readability and remove unneccesary functions that aren't utilized in the final training
- [ ] Provide example data for training-testing
- [ ] Find the issue with the model

# Setup

Install the necessary requirements by running

``` conda create -f hair.yaml ```

# Training

CUDA is required!!!

Run ```train.py -o``` for training. Please put your custom training data in a seperate ``` data ``` folder. Make sure that the data is compatible with ```GT3DDataset.py``` which is used for Dataloading.

# Licence

Parts of this code are Licenced under Attribution-NonCommercial 4.0 International since they contain code from: https://github.com/KeyuWu-CS/MonoHair which implements a similar approach for estimating growing direction.