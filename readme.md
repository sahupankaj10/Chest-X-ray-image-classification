# Chest X-ray images (Pneumonia and Normal) Classfification

Problem from: https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia

## About
Pneumonia is a very common disease. It can be either:
    1) Bacterial pneumonia
    2) Viral pneumonia
    3) Mycoplasma pneumonia and
    4) Fungal pneumonia.

This dataset consists pneumonia samples belonging to the first two classes. The dataset consists of only very few samples and that too unbalanced. The aim of this task is to develop a robust deep learning model from scratch on this limited amount of data.

In the current problem three types of images has been given - bacterial and viral Pneumonia and the Normal one. The task in this repository I was doing is to make a good classifier which perfectly classify the images into two category - Pneumonia(health diease) which coule be bacterial or viral and Normal.  

## Included:
- python notebook
- saved model 

## Use saved model

    new_model = tf.keras.models.load_model('xray-pneumonia-depthwise-convolution.h5')
    

# Result: Accuracy, Precion or Recall
The classes are imbalanced therefore validation accuracy won't be a good metric to analyze the model performance. The other metric, recall, precison and confusion matrix is good alternative to see the performance of the model. In this model, we got almost 98% recall value and 90% precison which is pretty good.

Also, here recall is most significant quantity even more than accuracy and precision because in this dataset we have to minimize the false negative as low as possible

False negative has to be intuitively minimized because falsely diagnosing a patient of pneumonia as not having a pneumonia is a much larger deal than falsely diagnosing a healthy person as a pneumonia patient which is our major concern . That is why we are making this model. To reduce the mistakes done by doctors accidentally.
