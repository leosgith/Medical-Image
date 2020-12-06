import os
from keras.preprocessing import image
import numpy as np
from pathlib import Path


# this is not a class, just add the main(): part to the end of your code or your main. And define the other two classes.
def main():

    paths = ['chest_xray/test/NORMAL', 'chest_xray/test/PNEUMONIA']
    # load all images into a list
    images = []
    for folder in paths:
        for img in os.listdir(folder):
            img = os.path.join(folder, img)
            #Just put the target size you used for training.
            img = image.load_img(img, target_size=(TARGET_SIZE, TARGET_SIZE))
            img = image.img_to_array(img)
            img = np.expand_dims(img, axis=0)
            images.append(img)

    # stack up images list to pass for prediction
    images = np.vstack(images)
    # classifier = our model
    prediction = np.argmax(classifier.predict(images), axis=-1)

    y_test = create_training_data(Path('chest_xray/test'))

    #function use and getting actual numbers

    tp, fp, tn ,fn, ac = perf_measure(y_test,prediction)

    precision = tp/(tp+fp)
    recall = tn/(tn+fn)
    f_score = (2*precision*recall)/(precision+recall)
    ac = ac/(tp+tn+fn+fp)

    print("Recall of the model is {:.2f}".format(recall))
    print("Precision of the model is {:.2f}".format(precision))
    print("F-Score is {:.2f}".format(f_score))
    print("Accuracy is {:.2f} %".format(ac*100))

def create_training_data(data_dir):
    #creating the training data
       #getting y_test once more, but dividing it into normal/ pneumonia (or 1/0) this time
    labels = ['NORMAL', 'PNEUMONIA']
    images = []

    for label in labels:
        dir = os.path.join(data_dir,label)
        class_num = labels.index(label)


        for image in os.listdir(dir):


            images.append([class_num])

    return np.array(images)


def perf_measure(test_batch, pred_final):
        TP = 0
        FP = 0
        TN = 0
        FN = 0

        for i in range(len(pred_final)):
           if test_batch[i]==pred_final[i]==1:
               TP += 1
            if test_batch[i]==1 and test_batch[i]!=pred_final[i]:
              FP += 1
            if test_batch[i]==pred_final[i]==0:
               TN += 1
            if test_batch[i]==0 and test_batch[i]!=pred_final[i]:
               FN += 1
        AC = TP + TN
        return(TP, FP, TN, FN, AC)
