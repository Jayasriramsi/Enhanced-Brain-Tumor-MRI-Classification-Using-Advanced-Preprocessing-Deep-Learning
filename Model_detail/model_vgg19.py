model_vgg_19 = tf.keras.applications.VGG19(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
)

#final model
model = Model(inputs = model_vgg_19.input, outputs = output)

x = model_vgg_19.output
#make the input of VGG19 model in higher dimension to single dimension
#x = Flatten()(x)
#fully connected layers
x = Dense(2028, activation='relu')(x)
#normalization
x = BatchNormalization()(x)
x = Dropout(0.3)(x)
x = Dense(1024, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)
output = Dense(4, activation='softmax')(x) 

# Freezing VGG19 convolutional layers stops them from updating.
# Useful if dataset is similar (transfer learning).
# But it limits fine-tuning lower layers for new or different features.

# Freeze the convolutional layers to prevent them from being trained
for layer in model_vgg_19.layers:
    layer.trainable = True

model.summary()

#from plot_model import plot_model
plot_model(model,to_file='model_vgg19.png', dpi=96, rankdir = 'TB', show_layer_names=True, show_shapes=True) 

#compile Model
model.compile(loss='categorical_crossentropy',
              optimizer =tf.keras.optimizers.Adam(learning_rate=5e-5),run_eagerly=True,
              metrics=["accuracy"])
    
#.h5 = Hierarchical Data Format Ver. 5 file, verbose =1, to see execution
tensorboard= TensorBoard(log_dir="logs")
checkpoint= ModelCheckpoint("vgg19_model.h5",
                           monitor="val_accuracy", verbose=1,
                           mode="auto", save_best_only=True)
#monitor: quantity to be monitored.     
#factor: factor by which the learning rate will be reduced. 
#patience: number of epochs with no improvement after which learning rate will be reduced.     
#verbose: int. 0: quiet, 1: update messages.
#min_delta: early stopping of epochs
#cooldown: number of epochs to wait before resuming normal operation after
reduce_lr=ReduceLROnPlateau(monitor="val_accuracy",
                           factor=0.3,
                           patience=2,verbose=1,
                           mode="auto") #lr dropping difference

# Calculate steps per epoch and validation steps
#train_steps = len(train_data_gen) // batch_size
#validation_steps = len(validation_data_gen) // batch_size
epochs = 30

history = model.fit(train_data_gen,
                    epochs=epochs,
                    validation_data=validation_data_gen,
                    callbacks=[tensorboard, checkpoint, reduce_lr]
                    )