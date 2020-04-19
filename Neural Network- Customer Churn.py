
import os

os.chdir("C:/Users/PC/Desktop/Deep Learning/Complete-Deep-Learning-master/ANN")

import pandas as pd

data=pd.read_csv("Churn_Modelling.CSV")

data.isna().sum()

X=data.iloc[:,3:13]
y=data.iloc[:,13]

geography=pd.get_dummies(X['Geography'])
gender=pd.get_dummies(X['Gender'])

X=pd.concat([X,geography,gender],axis=1)

X=X.drop(['Geography','Gender'],axis=1)

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y, test_size=0.2, random_state=0)



# Scaling

from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

import keras

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LeakyReLU, PReLU, ELU
from keras.layers import Dropout


classifier=Sequential()

# Adding input layer and first hidden layer

classifier.add(Dense(units = 6, kernel_initializer = 'he_uniform',activation='relu',input_dim = 13))

# Adding second hidden layer

classifier.add(Dense(units=6, kernel_initializer='he_uniform',activation='relu'))

# output layer

classifier.add(Dense(units=1,kernel_initializer='glorot_uniform',activation='sigmoid'))

# complile ANN

classifier.compile(optimizer='Adamax', loss='binary_crossentropy',metrics=['accuracy'])


# Fitting the ANN model to training set
model_hist=classifier.fit(X_train,y_train,batch_size=10,epochs=100, validation_split=.33)

print(model_hist.history.keys())

print(model_hist.history)

# summarize history for loss

import matplotlib.pyplot as plt

plt.plot(model_hist.history['accuracy'])
plt.plot(model_hist.history['val_accuracy'])

plt.plot(model_hist.history['loss'])
plt.plot(model_hist.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# Part 3 - Making the predictions and evaluating the model

y_pred=classifier.predict(X_test)
y_pred=(y_pred>0.5)

from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_pred,y_test)

print(cm)

from sklearn.metrics import accuracy_score

score=accuracy_score(y_pred,y_test)
print(score)

