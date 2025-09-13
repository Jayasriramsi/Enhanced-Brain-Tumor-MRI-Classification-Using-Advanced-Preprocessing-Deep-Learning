# Replace these with your actual column names
image_path_column = 'file'
category_id_column = 'category_id'
predicted_classes_column = 'predicted_classes'

# Filter misclassified samples and extract indices with corresponding image paths
misclassified_samples = target_and_predict[target_and_predict[category_id_column] != target_and_predict[predicted_classes_column]]
misclassified_indices_with_image_paths = misclassified_samples[[image_path_column, category_id_column, predicted_classes_column]]

print("Misclassified Samples with Image Paths:")
misclassified_indices_with_image_paths

# Summary
# Training: 99.90% accuracy, 0.004 loss
# Validation: 99.31% accuracy, 0.035 loss
# Testing: 99.69% accuracy
