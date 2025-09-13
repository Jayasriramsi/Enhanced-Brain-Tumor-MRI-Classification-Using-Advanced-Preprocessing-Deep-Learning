# Accuracy, Learning Rate And Loss Plot
def accuracy_lr_loss_plot():
    fig = plt.figure(figsize=(15,6))
    plt.subplot(131)
    plt.plot(history.history['accuracy'],'bo--', label = "accuracy")
    plt.plot(history.history['val_accuracy'], 'go--', label = "val_accuracy")
    plt.title("Training Data Accuracy Measurements")
    plt.xlabel("No of Epochs")
    plt.ylabel("Accuracy")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.subplot(132)
    plt.plot(history.history['lr'], 'bo--', label = 'learning Rate')
    plt.title("Learning Rate Measurements")
    plt.xlabel("No of Epochs")
    plt.ylabel("Learning Rate")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.subplot(133)
    plt.plot(history.history['loss'], 'bo--', label = 'loss')
    plt.plot(history.history['val_loss'], 'ro--', label = 'val_loss')
    plt.title("Training Data Loss Measurements")
    plt.xlabel("No of Epochs")
    plt.ylabel("Loss")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()
    accuracy_lr_loss_plot()

#  Model Prediction With Validation(Unseen Data)
def prepare_test_data(test_dataframe, x_columns, batch_size, image_size):  
    # Define parameters for ImageDataGenerator for train and validation
    test_image_data_generator = ImageDataGenerator(
        rescale=1./255  # Rescale pixel values to [0,1])
    test_generator = test_image_data_generator.flow_from_dataframe(
        batch_size=batch_size,
        dataframe=test_dataframe,
        shuffle=False,
        x_col=x_columns,
        y_col=None,
        target_size=(image_size, image_size),  # Set the target size for images
        class_mode=None  # For categorical labels)
    return test_generator
#testing columns details
testing.columns
#test images preprocessing
test_processed_images = prepare_test_data(validing,'file', 32, 224)

# Make predictions for images using the model
predictions = model.predict(test_processed_images, steps=len(testing) // 32 + 1)
predicted_classes = np.argmax(predictions, axis=1)
predicted_classes

#validing dataset to create new dataframe
target_and_predict = validing.copy()

#add column for predicted value
target_and_predict['predicted_classes'] = predicted_classes

#target_and_predict dataframe
target_and_predict


