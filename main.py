
import os
import cv2
import numpy as np
from keras.models import Sequential
from keras import layers, models
from sklearn.metrics import confusion_matrix , classification_report
import pickle

path = "Cervical Cancer Dataset/Train/"
# files = os.listdir(path)[:]
classes = { 0:"normal_columnar",1:"normal_intermediate",2:"normal_superficiel",3:"light_dysplastic",
            4:"moderate_dysplastic",5:"severe_dysplastic",6:"carcinoma_in_situ"}



trainX = []
testX = []
trainY = []
testY = []
for cl in classes:
    pth = path + classes[cl]
    print(pth)
    for img_name in os.listdir(pth):
        img = cv2.imread(pth + '/' + img_name)
        img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (50, 50))
        trainX.append(cv2.cvtColor(cv2.equalizeHist(img),cv2.COLOR_GRAY2BGR))
        trainY.append(cl)

path = "Cervical Cancer Dataset/Test/"
for cl in classes:
    pth = path + classes[cl]
    print(pth)
    for img_name in os.listdir(pth):
        img = cv2.imread(pth + '/' + img_name)
        img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (50, 50))
        testX.append(cv2.cvtColor(cv2.equalizeHist(img),cv2.COLOR_GRAY2BGR))
        testY.append(cl)


trainX = np.array(trainX)/255.0
trainY = np.array(trainY)
testX = np.array(testX)/255.0
testY = np.array(testY)
print(trainX.shape)
print(trainY.shape)
print(testX.shape)
print(testY.shape)

# cnn = Sequential([
#     layers.Conv2D(filters=50, kernel_size=(3, 3), activation='relu', input_shape=(50, 50, 3)),
#     layers.MaxPooling2D((2, 2)),
#
#     layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'),
#     layers.MaxPooling2D((2, 2)),
#
#     layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
#     layers.MaxPooling2D((2, 2)),
#
#     layers.Flatten(),
#     layers.Dense(64, activation='relu'),
#     layers.Dropout(0.3),
#     layers.Dense(7, activation='softmax')
# ])


cnn = Sequential([
    layers.Conv2D(filters=50, kernel_size=(3, 3), activation='relu', input_shape=(50, 50, 3)),
    # layers.MaxPooling2D((2, 2)),

    layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    # layers.MaxPooling2D((2, 2)),

    layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(filters=256, kernel_size=(3, 3), activation='relu'),
    # layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(7, activation='softmax')
])

cnn.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


cnn.fit(trainX, trainY,batch_size=128, epochs=30)

cnn.save('D:\PycharmProjects\Cervical Cancer Detection\cervical_model.h5')
y_pred = cnn.predict(testX)
y_pred_classes = [np.argmax(element) for element in y_pred]

print("Classification Report: \n", classification_report(testY, y_pred_classes))

pickle.dump(cnn,open('model.sav','wb'))