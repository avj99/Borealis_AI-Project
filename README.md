We started off by choosing a neural network. A Convolutional Neural Network (CNN)
is a Deep Learning algorithm that takes an input image and through its layers of convolution,
ReLu, and pooling, accurately classifies the image based on the classification label. We chose
CNN because CNN’s are specifically designed to process pixel data and are best for 2D
images. The traditional Neural Network known as Multilayer Perceptron (MLP) is not great at
image processing because it reacts poorly to shifting versions of an input image — it assumes
the object in the image will be in the same area at all times. CNN on the other hand, does not
pose this problem because it breaks down an image into multiple groups and it only focuses
on groups of pixels closest to each other.
Next, we chose the 3 different models we wanted to test: Simple Classifier, DenseNet,
and GoogLeNet. We hypothesized that having a pretrained and more complex model would
be better than a simpler model because It would have a higher accuracy and lower loss. For
the Simple Classifier model, we used the Pytorch Blitz tutorial for making a classifier since we
all didn’t have any previous experience. Our data, which consisted of images of the ASL
alphabet hand signs, was passed through this model, and after training the Simple Classifier
model it reached an accuracy of 7% and the loss decreased overtime, but the slope of this
function was not very steep and could definitely be improved. The second model we used
was DenseNet. We chose DenseNe because during our initial search to identify which models
would be best suited for image classification, DenseNet was a popular choice. Diving deeper
we found that some of the key traits of DenseNet are the emphasis on strengthening feature
propagation, and encouraging feature reuse — which is great since CNN emphasises on
manipulating features. However, things don’t always go as planned, and when we ran
DenseNet the results reflected poorly on our hypothesis that this model would yield a high
accuracy. The last model we used was GoogLeNet. We chose this model because of its 22
layers deep network, specifically designed for classification and object detection. This model
also won a Visual Recognition Competition in 2014, so we thought it would be interesting to
test this model with our dataset. GoogLeNet surpassed both previous models and had the
highest accuracy of approximately 14.85% with 11 epochs. Lastly, we played around with different hyperparameters and data augmentation to
see if the accuracy of our model could be improved. We first modified the learning rates of
all three of the models to try and find the most optimal for each. Learning rates are a hyper-parameter that affects how fast the algorithm learns, and whether or not the cost
function is minimized. Testing different learning rates turned out to be very useful and we
were able to achieve ~14% accuracy on GoogLeNet. The next hyperparameter we modified
was the number of epochs, or in other words, the number of times we iterate through all of
our data during training. Increasing the number of Epochs gives more insight into how the
model learns over time. Increasing epochs also shows when the accuracy plateaued
signifying when the model reached its optimal performance. Finally, we worked on some data
augmentation to see if it affected our results. We got all our data from Kaggle, but we
realized that the images were all very similar, and our accuracy was not very high. We
decided to apply multiple different augmentation techniques to the images in the dataset to
see if it helped our model learn better and improved our accuracy. Our results found that
there was not a big difference in our results when training our models with augmented vs
non-augmented data.
