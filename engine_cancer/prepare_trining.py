import random
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


from imutils import paths
from sklearn.model_selection import train_test_split
from keras.applications import DenseNet201
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import ImageDataGenerator
from keras import regularizers

from keras.applications.inception_v3 import InceptionV3 
from keras.optimizers import Adam
from keras.layers import GlobalAveragePooling2D,BatchNormalization
from keras.layers.core import Dense,Dropout
from keras.models import Model

from sklearn.metrics import confusion_matrix



class preapre:
    def __init__(self) -> None:
        pass
    def test(self):
        prepared_data_path = './tmp/prepared_test/'
        prepared_data_list_filenames = (sorted(list(paths.list_images(prepared_data_path))))

        random.shuffle(prepared_data_list_filenames)
        prepared_data_list_labels = []

        for line in prepared_data_list_filenames:
            prepared_data_list_labels.append(line.split(os.path.sep)[3])

        I_series = pd.Series(prepared_data_list_filenames, name='filenames')
        L_series = pd.Series(prepared_data_list_labels, name='labels')
        test_df = pd.concat( [I_series, L_series], axis=1) 

        # print('-- test Datafarame --')
        # print(test_df.head())
        # print number of each class:        
        a=test_df['labels'].value_counts()
        # print(a)
        return test_df

    def data(self):
        prepared_data_path = './tmp/prepared_data/'
        prepared_data_list_filenames = (sorted(list(paths.list_images(prepared_data_path))))

        random.shuffle(prepared_data_list_filenames)
        prepared_data_list_labels = []

        for line in prepared_data_list_filenames:
            prepared_data_list_labels.append(line.split(os.path.sep)[3])

        I_series = pd.Series(prepared_data_list_filenames, name='filenames')
        L_series = pd.Series(prepared_data_list_labels, name='labels')
        df = pd.concat( [I_series, L_series], axis=1) 

        # print('-- train/valid Datafarame --')

        # print(df.head())
        # print number of each class:        
        a=df['labels'].value_counts()
        # print(a)

        return df
    
    def training(self):
        df = self.data()
        test_df = self.test()
        SPLIT= 0.90

        TRAIN_DF, VALID_DF = train_test_split(df, train_size=SPLIT, shuffle=True, random_state=88)

        # print('Train samples: ', len(TRAIN_DF))
        # print('Valid samples: ', len(VALID_DF))
        # print('Test samples : ', len(test_df))
        # print(test_df['labels'].value_counts())
        
        BATCH_SIZE= 32
        IMG_SHAPE= (224, 224, 3)
        IMG_SIZE= (224, 224)
        
        gen = ImageDataGenerator(rescale=1./255, 
                                 vertical_flip=True,
                                 horizontal_flip=True)
                                 #rotation_range=10)
        
        gen2 = ImageDataGenerator(rescale=1./255)
        
        train_gen = gen.flow_from_dataframe(TRAIN_DF,
                                            x_col= 'filenames',
                                            y_col= 'labels',
                                            target_size= IMG_SIZE,
                                            class_mode= 'categorical',
                                            color_mode= 'rgb',
                                            shuffle= True,
                                            batch_size= BATCH_SIZE,
                                            seed=88
        )
        
        valid_gen= gen2.flow_from_dataframe(VALID_DF,
                                            x_col= 'filenames',
                                            y_col= 'labels',
                                            target_size= IMG_SIZE,
                                            class_mode= 'categorical',
                                            color_mode= 'rgb',
                                            shuffle= True,
                                            batch_size= BATCH_SIZE,
                                            seed=88
        )
        test_gen= gen2.flow_from_dataframe(test_df,
                                           x_col= 'filenames',
                                           y_col= 'labels',
                                           target_size= IMG_SIZE,
                                           class_mode= 'categorical',
                                           color_mode= 'rgb',
                                           shuffle= True,
                                           batch_size= 325,
                                           seed=88
        )
        
        
        STEPS= int(len(train_gen.labels)/BATCH_SIZE)
        # print(STEPS)

        genn=train_gen
        class_dictionary= genn.class_indices
        class_names = list(class_dictionary.keys())
        images, labels = next(genn) #get sample batch from the generator
        plt.figure(figsize=(20,20))
        length = len(labels)

        if length<25:
            r=length
        else:
            r=25

        for i in range(r):
            plt.subplot(5, 5, i+1)
            image= (images[i])
            #image= cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            plt.imshow(image)
            index=np.argmax(labels[i])
            class_name= class_names[index]
            plt.title(class_name, color='blue', fontsize=16)
            plt.axis('off')
        # plt.show()

        # print(train_gen)
        base_model= tf.keras.applications.mobilenet_v2.MobileNetV2(include_top=False,weights='imagenet',input_shape=(224,224,3))
        #tf.keras.applications.MobileNetV3Small

        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = BatchNormalization()(x) 
        #x = Dense(128, activation= 'relu',kernel_initializer='he_uniform')(x)
        #x = BatchNormalization()(x)
        #x = Dropout(0.3)(x)
        predictions = Dense(4, activation= "softmax")(x) 
        model = Model(inputs=base_model.input, outputs=predictions) 

        # return test_gen, valid_gen

        initial_learning_rate = 0.0001
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate,
            decay_steps=40,
            decay_rate=0.96,
            staircase=False)


        model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(learning_rate=lr_schedule, beta_1=0.9, beta_2=0.999, epsilon=1e-08), #optimizer=Adam(learning_rate=0.00001,decay = 10e-5),
                      metrics=['accuracy'])
                     # option = run_opts)



        model.fit(x=train_gen,
                    epochs=30,
                    validation_data=valid_gen,
                    steps_per_epoch=None,
                    workers=2
                )

        
        acc=model.evaluate(test_gen,batch_size=32, steps=None, verbose=1)[1]*100
        msg='Model accuracy on test set: ' + str(acc)
        # print(msg, (0,255,0), (55,65,80))
        for X_batch, y_batch in test_gen:
            y_test = y_batch
            X_test = X_batch
            break

        print('test label shape',y_test.shape)
        print('test image shape',X_test.shape)
        print('Evaluate on test-data:')
        model.evaluate(X_test,y_test)

        pred = model.predict(X_test)

        bin_predict = np.argmax(pred,axis=1)
        y_test = np.argmax(y_test,axis=1)


        #Confusion matrix:
        matrix = confusion_matrix(y_test, bin_predict)
        print('Confusion Matrix:\n',matrix)

        preds = model.predict(X_test)
        #print(preds)
        print('Shape of preds: ', preds.shape)
        plt.figure(figsize = (12, 12))

        number = np.random.choice(preds.shape[0])

        for i in range(25):
            plt.subplot(5, 5, i + 1)
            plt.grid(False)
            plt.xticks([])
            plt.yticks([])
            number = np.random.choice(preds.shape[0])
            pred = np.argmax(preds[number])
            actual = (y_test[number])
            col = 'g'
            if pred != actual:
                col = 'r'
            plt.xlabel('N={} | P={} | GT={}'.format(number, pred, actual), color = col) #N= number P= prediction GT= actual (ground truth)
            image= X_test[number]#cv2.cvtColor(X_test[number], cv2.COLOR_BGR2RGB)
            plt.imshow(((image* 255).astype(np.uint8)), cmap='binary')
        # plt.show()


        from tensorflow import lite
        converter = lite.TFLiteConverter.from_keras_model(model)
        tfmodel = converter.convert()
        open ('./model4/model5.tflite','wb').write(tfmodel)




if __name__ == "__main__":
    pre = preapre()
    # pre.test()
    # pre.data()
    pre.training()