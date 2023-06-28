# ECG-pretraining
Deep Learning project that aims at building a robust pre-trained model in a self-supervised fashion with transferable capabilities when dealing with 12-lead ECGs interpretation.
## Training
The training was conducted in a self-supervised manner and was inspired by `BERT`'s (`RoBERTA`) pre-training: masked language modelling and masked tokens' reconstruction loss. 
Here, a 12-lead ECG is organized in a 12 x `window` matrix, where window represents tracings' duration (or tracings sampling's duration), and `patch_height` x `patch_width` shaped patches are extacted from it. Some of these patches are then randomly selected for a masking procedure, that is a substitution of the patches with a special learnable "`mask patch`". The sequence of patches, masked and not, are then fed into the network.
## Architecture
The network architecture is strongly based upon the `Transformer's encoder`, which is preceded by an `embedder` that pre-processes the input patches. The encoder is followed by a `decoder` whose goal is to reconstruct the input patches from the representation coming out from the encoder
