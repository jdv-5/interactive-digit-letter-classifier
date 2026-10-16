## Canvases of nums and letters

 There are 2 main files (mnist_letters and mnist_nums) where each one have 2 models to predict; a Neural Network and a Softmax Regression, every model have a column of probabilities for each number or letter,
 in the case of the letters there will be the 10 letters with more probabilities based on the models, and in the case of the numbers there are the 0-9 numbers probabilities, 
 the highest probability will be colored with red.

### Controls:

- **C**: Delete the entire canvas
- **Enter**: Change the image with a random image from the test data
- **Left Click**: Write over the canvas
- **Right Click**: Delete over the canvas


 ## Training files

 There are 2 files for to train the models of each canvas, in the files there are 2 main methods one for train the Neural Network and other for train the Softmax Regression 
each method have as parameters: epochs and learning rate, you also can edit the architecture of the Neural Network the length of the minibatch or change to all the batch, same for the Softmax Regression
