# Evaluate model on training data
train_score = model.evaluate(train_data_gen)
# Print results clearly
print("\nTraining Loss: {:.3f} | Training Accuracy: {:.3f}%"
      .format(train_score[0], train_score[1] * 100))

# We have training loss 0.004% and training accuracy 99.905% after evaluation.
# 📌We have training loss 0.004% and training accuracy 99.905% after evaluation.


# Evaluate model on validation data
val_score = model.evaluate(validation_data_gen)

# Print results clearly
print("\nValidation Loss: {:.3f} | Validation Accuracy: {:.3f}%"
      .format(val_score[0], val_score[1] * 100))

# We have validation loss 0.035 and validation accuracy 99.314% after evaluation.
# 📌We have validation loss 0.035 and validation accuracy 99.314% after evaluation.

# Testing Accuracy

# Calculate accuracy
accuracy = accuracy_score(target_and_predict['category_id'], target_and_predict['predicted_classes'])
print(f"\nWe have test accuracy {accuracy * 100:.4f}% after test images evaluation.")
# We have test accuracy 99.6947% after test images evaluation.
# 📌We have test accuracy 99.6947% after test images evaluation.