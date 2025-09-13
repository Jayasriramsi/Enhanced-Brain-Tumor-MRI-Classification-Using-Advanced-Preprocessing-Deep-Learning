def data_preparation(data, subpath_dir, dataset_dir):
    data_list = []  # List to collect file paths, category IDs, and category names
    # Iterate over all categories (glioma, meningioma, notumor, pituitary)
    for category_id, category in enumerate(CATEGORIES):
        # Path to category folder inside dataset directory
        category_dir = os.path.join(dataset_dir, category)
        # Loop through all image files in this category
        for file in os.listdir(category_dir):
            # Append file path, category ID, and category name
            data_list.append([
                f'{data_dir}/{subpath_dir}/{category}/{file}',  # file path
                category_id,                                   # category numeric ID
                category                                       # category name
            ])
    
    # Convert list to DataFrame
    data = pd.DataFrame(data_list, columns=['file', 'category_id', 'category'])
    # Print dataset shape for quick check
    print(f"Shape of {subpath_dir} dataset: {data.shape}")
    return data

# Create an empty DataFrame (placeholder, will be filled in function)
train = pd.DataFrame()

# Prepare the Training dataset dataframe using the helper function
trainset = data_preparation(train, 'Training', train_dir)

# Display first 2 rows to check the structure of the dataset
trainset.head(2)

# Access the first file path from the training dataset
trainset.iloc[0]['file']

# Create Testing dataset DataFrame
test = pd.DataFrame()
testset = data_preparation(test, 'Testing', test_dir)

# Display first 2 rows of Testing dataset
testset.head(2)

# Create Validation dataset DataFrame
valid = pd.DataFrame()
validset = data_preparation(valid, 'Validation', valid_dir)

# Display first 2 rows of Validation dataset
validset.head(2)

# Check the distribution of images per category in Training dataset
trainset['category'].value_counts()

#Sampling parameters
SAMPLE_PER_CATEGORIES = 1321   # Number of samples per category (for balancing)
SEED = 42                      # Random seed for reproducibility
WIDTH = 150                    # Image width after resizing
HEIGHT = 150                   # Image height after resizing
DEPTH = 3                      # Number of channels (RGB)

# Input shape for CNN (to be used in VGG19)
INPUT_SHAPE = (WIDTH, HEIGHT, DEPTH)

# 📌 TRAINING DATA PREPROCESSING
# Make a copy of training dataset
training = trainset.copy()
# Balance the dataset by taking equal number of samples per category
training = pd.concat([
    training[training['category'] == c][:SAMPLE_PER_CATEGORIES] 
    for c in CATEGORIES
])

# Shuffle the dataset
training = training.sample(frac=1, random_state=SEED)
# Reset index after shuffling
training.index = np.arange(len(training))
# Check the final shape of training dataset
training.shape
# Display first 10 samples of balanced training dataset
training.head(10)

# 📌 VALIDATION DATA PREPROCESSING
validing = validset.copy()
validing.head(5)   # Show first 5 rows of validation dataset

# 📌 TESTING DATA PREPROCESSING
testing = testset.copy()
testing.head(5)    # Show first 5 rows of testing dataset

