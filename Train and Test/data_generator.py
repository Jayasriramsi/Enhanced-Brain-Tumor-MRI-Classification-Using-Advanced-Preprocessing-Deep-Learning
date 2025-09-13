# Path to the image you want to augment
image_path = 'MRI Image Dataset for Brain Tumor/Training/glioma/Tr-gl_0010.jpg'  # Replace with the actual path to your image

# Load the image
img = load_img(image_path)
img_array = img_to_array(img)
img_array = img_array.reshape((1,) + img_array.shape)  # Reshape to (1, height, width, channels)

# Define ImageDataGenerators with different augmentation settings
datagen_rotation = ImageDataGenerator(rotation_range=45)
datagen_width_shift = ImageDataGenerator(width_shift_range=0.6)
datagen_height_shift = ImageDataGenerator(height_shift_range=0.4)
datagen_shear = ImageDataGenerator(shear_range=2)
datagen_flip = ImageDataGenerator(horizontal_flip=True)

data_generators = [datagen_rotation, datagen_shear, datagen_width_shift, datagen_height_shift, datagen_flip]
attributes = ['Rotation', 'Shear', 'Width Shift', 'Height Shift',  'Horizontal Flip']

for i, datagen in enumerate(data_generators):
    augmented_images = []
    plt.figure(figsize=(8, 8))
    plt.suptitle(attributes[i], fontsize=16)
    
    generator = datagen.flow(img_array, batch_size=1)
    for j in range(6):
        batch = generator.next()
        augmented_images.append(batch[0].astype('uint8'))
        plt.subplot(3, 3, j + 1)
        plt.imshow(batch[0].astype('uint8'))
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()