# Replace these with your actual column names
category_id_column = 'category_id'
predicted_classes_column = 'predicted_classes'
category_name_column = 'category'

# Create a mapping dictionary from category_id to category_name
id_to_name_mapping = dict(zip(target_and_predict[category_id_column], target_and_predict[category_name_column]))
print(f"id_to_name_mapping:\n{id_to_name_mapping}")
# Replace numeric values in 'category_id' and 'predicted_classes' columns with their names
target_and_predict['category_id'] = target_and_predict['category_id'].map(id_to_name_mapping)
target_and_predict['predicted_classes'] = target_and_predict['predicted_classes'].map(id_to_name_mapping)

# Generate classification report with names instead of numeric values
report = classification_report(
    target_and_predict['category_id'],
    target_and_predict['predicted_classes']
)

print(report)

# Generate confusion matrix
conf_matrix = confusion_matrix(
    target_and_predict['category_id'],
    target_and_predict['predicted_classes']
)

# Plot confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=list(id_to_name_mapping.values()))
fig, ax = plt.subplots(figsize=(8, 8))
disp.plot(cmap='ocean_r',ax=ax,xticks_rotation=60)
plt.show()

