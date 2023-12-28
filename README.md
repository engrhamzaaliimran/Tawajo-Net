# Tawajo Net: Human Activity Recognition Neural Network

This repository contains the implementation of Tawajo Net, a neural network designed for Human Activity Recognition (HAR). The model integrates bidirectional GRU layers, attention mechanisms, and various convolutional layers to effectively capture intricate patterns in sequential data. Tawajo Net has been trained and evaluated on the Human Activity Recognition dataset by Wisdom Lab. The results of this study are detailed in an article currently under review at an IEEE journal.

## Network Architecture

The Tawajo Net architecture, implemented in the script as the function `create_model(trainX, trainy)`, is tailored for HAR tasks. Here's an overview of the architecture:

1. **Input Layer:**
   - Shape: `(n_timesteps, n_features)`
   - Name: 'data'

2. **Bidirectional GRU Layer:**
   - 64 units
   - Return sequences

3. **Attention Mechanism:**
   - Applied to the output of the GRU layer
   - Utilizes a tanh activation followed by a softmax activation
   - Multiplies the attention weights with the GRU layer output

4. **Convolutional Blocks:**
   - Three convolutional blocks with different kernel sizes (1x3, 1x5, 1x7)
   - Each block includes two convolutional layers with 64 filters
   - Additional 1x1 convolution directly on the output of the attention mechanism
   - MaxPooling followed by 1x1 convolution for downsampling

5. **Concatenation:**
   - Concatenates the outputs of all convolutional blocks
   - Reduces dimensionality with a 1x1 convolution

6. **Dropout Layer:**
   - Dropout with a rate of 0.5 applied to the concatenated features

7. **Additional Convolutional Layers:**
   - Two convolutional layers with 128 and 64 filters respectively

8. **Global Average Pooling:**
   - Applied to the output of the last convolutional layer

9. **Output Layer:**
   - Dense layer with softmax activation
   - Outputs the final predictions

## Model Summary

The detailed model summary provides insights into the layers and their configurations.

## Usage

To use Tawajo Net, instantiate it by calling `create_model(trainX, trainy)`, where `trainX` is the input training data and `trainy` is the corresponding target labels. The model can then be compiled and trained using standard Keras functions.

```python
tawajo_net = create_model(trainX, trainy)
tawajo_net.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
tawajo_net.fit(trainX, trainy, epochs=10, batch_size=32, validation_data=(valX, valy))```

Feel free to customize the model architecture and parameters according to your specific HAR requirements.

## Dependencies
TensorFlow (>=2.0)
Keras (>=2.0)

## Article Under Review
The results of Tawajo Net on the Human Activity Recognition dataset by Wisdom Lab are detailed in an article currently under review at an IEEE journal.

## License
This project is licensed under the MIT License - see the LICENSE.md file for details.
