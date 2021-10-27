# Automated detection of different emotions from textual comments and feedbacks

## Aim:
To develop a deep learning algorithm to detect different
types of emotions contained in a collection of english
sentences or a large paragraph.

## Synopsis:
Social networking platforms have become an essential means for communicating
feelings to the entire world. The detection of text emotions is a content-based
classification problem. Emotion detection is the extraction and analyzing various
emotions from textual data. In some applications, sentiment analysis is insufficient
and hence requires emotion detection, which determines an individual’s
emotional/mental state precisely. Hence, developing an algorithm to understand
and detect emotions from textual data can be beneficial for various industries.

## Solution approach:
Deep learning algorithms such as Long Short Term Memory(LSTM) can be used
for the task. LSTM has the property of selectively remembering patterns for long
durations of time. The dataset used for the purpose is Emotion Dataset for Emotion
Recognition Tasks. On feeding the model with input text, it will try to predict the
emotion expressed, whether it’s joy, sadness, love, anger, fear, or surprise.

## Assumption:
The dataset consists of Twitter messages with six basic emotions: anger, fear, joy,
sadness, and surprise. The dataset is distinctly separated into train, test, and
validation sets. The training dataset consists of 16,000 tweets with corresponding
labels. The test and validation dataset both consist of 2000 tweets. It’s a
multi-class classification problem with the target emotions like anger, fear, joy,
sadness, and surprise.
The dataset is not preprocessed. The messages are being preprocessed and
tokenized using different Python libraries and algorithms before feeding them into
the model.

## Project Diagrams:
![image of project flow](https://i.ibb.co/JkSn1k8/project-nlp.png)

## Model architecture:
![model](https://i.ibb.co/Rj55FBN/model.png)

## Algorithm:
Multi-layer Bidirectional Long Short Term Memory(LSTM)

## Outcome:
For the following statement:

*“Everyone keeps telling me that time heals all wounds, but no one can tell me what I’m supposed
to do right now. Right now I can’t sleep. It’s right now that I can’t eat. Right now I still hear his
voice and sense his presence even though I know he’s not here. Right now all I seem to do is cry.
I know all about time and wounds healing, but even if I had all the time in the world, I still don’t
know what to do with all this hurt right now.”* <br>
The output is sadness.

*“Giving away something that was of great benefit or of requirement to the receiver definitely
brings in a feel of happiness and fulfillment. No matter whatever situation you may be in, when
you pass out things that are of great help and happiness to others, you too feel the same.”* <br>
The output is joy.

*“Dogs are friendly and they love human companionship. Whose ego would not be gratified at the
sight of a happy dog who can't wait to greet you at the end of a hard day? Your dog waits for you
by the door, face smiling, mouth open and tail wagging, ready to dote on you, his best friend in
the world.”* <br>
The output is love.
  
*“He well remembered the last interview he had had with the old prince at the time of the
enrollment, when in reply to an invitation to dinner he had had to listen to an angry
reprimand for not having provided his full quota of men.”* <br>
The output is anger.
  
*“My childhood fear at the time unfortunately was scary movies, heights, and being afraid of the
dark. As a young child, mainly around the age of 4 to 7 years old, screams and loud stressful
noises wasn’t generally my favorite amusement.”* <br>
The output is fear.
  
*“In my birthday my father gave me a pleasant surprise that is cycle. The sudden surpise of my
father is shocking to me. I was very happy about what my father had gave me a nice gift on my
birthday and my friends also wished me.”* <br>
The output is joy/surprise.

## Detailed Presentation with proof of reasonable accuracy:
The texts in the dataset are of different lengths. So, the padded sequence is used
to zero pad the sequence. The training dataset is divided into batches with
BATCH_SIZE = 32.

MODEL: tf.keras.Sequential model(7 layers used)
1. The first layer is an Embedding layer. The output of the Embedding layer is a
2D vector with one embedding for each word in the input sequence of words
(input document). This layer is trainable and after training, words that are
similar have similar vector representation.
2. The second and third layer is Bi-directional LSTM. These are RNN capable of
learning order dependence in sequence prediction problems. The input
propagates forward and backward through this network. This helps to realize
the emotions of the text.
3. The fourth and sixth is the dropout layer. It randomly sets input units to 0 with a
frequency of rate at each step during training time, which helps prevent
overfitting.
4. The fixed-length output vector from the previous layer is piped through a fully
connected dense layer with 64 hidden units and uses rectified linear activation
function.
5. The final layer is also a fully connected dense layer with 6 output nodes. This
uses the softmax activation function. The 6 output nodes are used to find the
probability of the six basic types of emotions.

COMPILING THE MODEL:<br>
For the training of the model, a loss function and an optimizer are required. Since
this is a multiclass classification problem, a categorical cross-entropy loss function
is used. And for optimization purposes, adam optimizer has been used.
The model is trained with the above parameters.<br>
The model.fit result with epochs=5 gives the following accuracy:<br>
Epoch 1/5<br>
500/500 [==============================] - 86s 153ms/step - loss: 1.0692 - accuracy: 0.5719 - val_loss: 0.5342 - val_accuracy: 0.7965<br>
Epoch 2/5<br>
500/500 [==============================] - 75s 150ms/step - loss: 0.3831 - accuracy: 0.8683 - val_loss: 0.2726 - val_accuracy: 0.9030<br>
Epoch 3/5<br>
500/500 [==============================] - 75s 149ms/step - loss: 0.2002 - accuracy: 0.9279 - val_loss: 0.2060 - val_accuracy: 0.9165<br>
Epoch 4/5<br>
500/500 [==============================] - 75s 150ms/step - loss: 0.1292 - accuracy: 0.9522 - val_loss: 0.2256 - val_accuracy: 0.9140<br>
Epoch 5/5<br>
500/500 [==============================] - 75s 150ms/step - loss: 0.0990 - accuracy: 0.9622 - val_loss: 0.2299 - val_accuracy: 0.9150<br>

FINAL ACCURACY:<br>
63/63 [==============================] - 3s 51ms/step - loss: 0.2299 - accuracy: 0.9150 <br>
Test Loss: 0.22988361120224
Test Accuracy: 0.9150000214576721

## Graphs:
![graph](https://i.ibb.co/kX8fKYY/graph.png)

## Exception considered:

Accuracy and loss vary slightly on every compilation of the model. This may give a
slight difference in the prediction value for a surprise emotion.
For smaller sentences, the model achieves better accuracy without padding,
however, for paragraphs, it does not make much difference.

## Enhancement Scope:

Increasing the model accuracy without over-fitting can be the enhancement scope.
This can be achieved by adjusting the layers and parameters. The number of
epochs with respect to BATCH_SIZE can also vary the accuracy


