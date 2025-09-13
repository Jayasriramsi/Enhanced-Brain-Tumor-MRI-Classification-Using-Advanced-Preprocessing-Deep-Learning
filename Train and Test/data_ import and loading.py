# Base directory where the dataset is stored
data_dir = "MRI Image Dataset for Brain Tumor"

# Sub-directories for Training, Testing, and Validation datasets
train_dir = os.path.join(data_dir, "Training")
test_dir  = os.path.join(data_dir, "Testing")
valid_dir = os.path.join(data_dir, "Validation")

# Category names based on sub-folder names in the dataset
CATEGORIES = ["glioma", "meningioma", "notumor", "pituitary"]
# Total number of categories (useful for model output layer)
NUM_OF_CATEGORIES = len(CATEGORIES)
# Print number of categories
NUM_OF_CATEGORIES

# Data Information
def data_information(sub_data_dir):
    for category in CATEGORIES:
        print('{} {} images'.format(category,len(os.listdir(os.path.join(sub_data_dir, category)))))

#  Display dataset summary for Training set
print("\n Training Dataset: \n")
data_information(train_dir)

# Display dataset summary for Testing set
print("\n Testing Dataset: \n")
data_information(test_dir)

# Display dataset summary for Validation set
print("\n Validation Dataset: \n")
data_information(valid_dir)
