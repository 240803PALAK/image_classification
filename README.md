# image_classification
Image Classification Using CNN


    import tensorflow as tf
    from tensorflow.keras import datasets,layers,models
    import numpy as np
    import matplotlib.pyplot as plt
    (X_train,y_train),(X_test,y_test)=datasets.cifar10.load_data()
    y_train=y_train.reshape(-1,)
    y_train[:5]
    classes=['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
    def plot_sample(X,y,index):
        plt.figure(figsize=(15,2))
        plt.imshow(X[index])
        class_name = classes[int(y[index])]
        plt.xlabel(class_name)
    plot_sample(X_train,y_train,5)
    plot_sample(X_train,y_train,120)
    
    
    X_train=X_train/255.0
    X_test=X_test/255.0
    ann = tf.keras.models.Sequential([
        layers.Flatten(input_shape=(32,32,3)),
        layers.Dense(3000,activation='relu'),
        layers.Dense(1000,activation='relu'),
        layers.Dense(10,activation='softmax')
    ])
    
    # Compile the model
    ann.compile(optimizer='SGD',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
    
    # Train the model
    ann.fit(X_train, y_train, epochs=5, batch_size=32)
    
    from sklearn.metrics import confusion_matrix,classification_report
    import numpy as np
    y_pred=ann.predict(X_test)
    y_pred_classes=[np.argmax(element) for element in y_pred]
    print('Classification report: \n',classification_report(y_test,y_pred_classes))
    import seaborn as sns
    plt.figure(figsize=(14,7))
    sns.heatmap(y_pred,annot=True)
    plt.ylabel('Truth')
    plt.xlabel('Prediction')
    plt.title('Confusion matrix')
    plt.show()
    cnn=tf.keras.models.Sequential([
        layers.Conv2D(filters=32,kernel_size=(3,3),activation='relu',input_shape=(32,32,3)),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(filters=64,kernel_size=(3,3),activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Flatten(),
        layers.Dense(64,activation='relu'),
        layers.Dense(10,activation='softmax')
    ])
    cnn.compile(optimizer='adam',
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])
    cnn.fit(X_train,y_train,epochs=10)
    cnn.evaluate(X_test,y_test)
    y_pred=cnn.predict(X_test)
    y_pred[:5]
    y_classes=[np.argmax(element) for element in y_pred]
    y_classes[:5]
    y_test[:5]
    plot_sample(X_test,y_test,60)
    plot_sample(X_test,y_test,100)
    classes[y_classes[60]]

