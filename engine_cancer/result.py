import matplotlib.pyplot as plt
import prepare_trining
import numpy as np

from sklearn.metrics import confusion_matrix

def tr_plot(tr_data, start_epoch):
    #Plot the training and validation data
    tacc=tr_data.history['accuracy']
    tloss=tr_data.history['loss']
    vacc=tr_data.history['val_accuracy']
    vloss=tr_data.history['val_loss']
    Epoch_count=len(tacc)+ start_epoch
    Epochs=[]
    for i in range (start_epoch ,Epoch_count):
        Epochs.append(i+1)   
    index_loss=np.argmin(vloss)#  this is the epoch with the lowest validation loss
    val_lowest=vloss[index_loss]
    index_acc=np.argmax(vacc)
    acc_highest=vacc[index_acc]
    plt.style.use('fivethirtyeight')
    sc_label='best epoch= '+ str(index_loss+1 +start_epoch)
    vc_label='best epoch= '+ str(index_acc + 1+ start_epoch)
    fig,axes=plt.subplots(nrows=1, ncols=2, figsize=(20,8))
    axes[0].plot(Epochs,tloss, 'r', label='Training loss')
    axes[0].plot(Epochs,vloss,'g',label='Validation loss' )
    axes[0].scatter(index_loss+1 +start_epoch,val_lowest, s=150, c= 'blue', label=sc_label)
    axes[0].set_title('Training and Validation Loss')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[1].plot (Epochs,tacc,'r',label= 'Training Accuracy')
    axes[1].plot (Epochs,vacc,'g',label= 'Validation Accuracy')
    axes[1].scatter(index_acc+1 +start_epoch,acc_highest, s=150, c= 'blue', label=vc_label)
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    plt.tight_layout
    #plt.style.use('fivethirtyeight')
    plt.show()



pre = prepare_trining.preapre()
# history = pre.training()[2]
test_gen = pre.training()[1]
model= pre.training()[0]

print(test_gen)
# tr_plot(history, 0)
# acc=model.evaluate(test_gen,batch_size=32, steps=None, verbose=1)[1]*100
# msg='Model accuracy on test set: ' + str(acc)
# print(msg, (0,255,0), (55,65,80))

# for X_batch, y_batch in test_gen:
#     y_test = y_batch
#     X_test = X_batch
#     break
    
# print('test label shape',y_test.shape)
# print('test image shape',X_test.shape)
# print('Evaluate on test-data:')
# model.evaluate(X_test,y_test)

# pred = model.predict(X_test)

# bin_predict = np.argmax(pred,axis=1)
# y_test = np.argmax(y_test,axis=1)


# #Confusion matrix:
# matrix = confusion_matrix(y_test, bin_predict)
# print('Confusion Matrix:\n',matrix)

# preds = model.predict(X_test)
# #print(preds)
# print('Shape of preds: ', preds.shape)
# plt.figure(figsize = (12, 12))

# number = np.random.choice(preds.shape[0])

# for i in range(25):
#     plt.subplot(5, 5, i + 1)
#     plt.grid(False)
#     plt.xticks([])
#     plt.yticks([])
#     number = np.random.choice(preds.shape[0])
#     pred = np.argmax(preds[number])
#     actual = (y_test[number])
#     col = 'g'
#     if pred != actual:
#         col = 'r'
#     plt.xlabel('N={} | P={} | GT={}'.format(number, pred, actual), color = col) #N= number P= prediction GT= actual (ground truth)
#     image= X_test[number]#cv2.cvtColor(X_test[number], cv2.COLOR_BGR2RGB)
#     plt.imshow(((image* 255).astype(np.uint8)), cmap='binary')
# plt.show()