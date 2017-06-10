import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, normalization

batch_size = 128
num_classes = 10
epochs = 2000

(x_train,y_train),(x_test,y_test) = cifar10.load_data()

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()

model.add(Conv2D(16,(3,3),padding='same', input_shape=x_train.shape[1:],kernel_initializer='glorot_normal', bias_initializer=keras.initializers.Constant(0.1)))
model.add(normalization.BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(32,(3,3),padding='same',kernel_initializer='glorot_normal', bias_initializer=keras.initializers.Constant(0.1)))
model.add(normalization.BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3,3),strides=2,padding='valid'))
model.add(Dropout(0.2))

model.add(Conv2D(32,(3,3),padding='same',kernel_initializer='glorot_normal', bias_initializer=keras.initializers.Constant(0.1)))
model.add(normalization.BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(64,(3,3),padding='same',kernel_initializer='glorot_normal', bias_initializer=keras.initializers.Constant(0.1)))
model.add(normalization.BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3,3),strides=2,padding='valid'))
model.add(Dropout(0.2))

model.add(Conv2D(64,(3,3),padding='same',kernel_initializer='glorot_normal', bias_initializer=keras.initializers.Constant(0.1)))
model.add(normalization.BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(128,(5,5),padding='same',kernel_initializer='glorot_normal', bias_initializer=keras.initializers.Constant(0.1)))
model.add(normalization.BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2),padding='valid'))
model.add(Dropout(0.2))

model.add(Conv2D(128,(5,5),padding='same',kernel_initializer='glorot_normal', bias_initializer=keras.initializers.Constant(0.1)))
model.add(normalization.BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(256,(5,5),padding='same',kernel_initializer='glorot_normal', bias_initializer=keras.initializers.Constant(0.1)))
model.add(normalization.BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2),padding='valid'))
model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(128,kernel_initializer='glorot_normal', bias_initializer=keras.initializers.Constant(0.1)))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(num_classes))
model.add(Activation('softmax'))

opt = keras.optimizers.SGD(lr=0.05,decay=0.001, momentum=0.9)
#opt = keras.optimizers.Adam(0.001)
#opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])

x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

datagen = ImageDataGenerator(rotation_range=90,width_shift_range=0.1,height_shift_range=0.1,horizontal_flip=True)

datagen.fit(x_train)

model.fit_generator(datagen.flow(x_train,y_train,batch_size=batch_size),steps_per_epoch=x_train.shape[0]//batch_size,epochs=epochs,validation_data=(x_test,y_test))