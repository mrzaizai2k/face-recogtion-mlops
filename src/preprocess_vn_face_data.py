import os
import pandas as pd
import cv2

# Define paths
csv_file_path = 'data/vn_celeb_face_recognition/train.csv'
train_folder_path = 'data/vn_celeb_face_recognition/train/'
destination_folder_path = 'data/facenet_vn_cropped/'
min_images_folder =2

# Create destination folder if it does not exist
if not os.path.exists(destination_folder_path):
    os.makedirs(destination_folder_path)

# Read CSV file
data = pd.read_csv(csv_file_path)

# Count the number of images per label
label_counts = data['label'].value_counts()

# Filter labels that have at least 2 images
valid_labels = label_counts[label_counts >= min_images_folder].index

# Filter data to include only rows with valid labels
filtered_data = data[data['label'].isin(valid_labels)]

# Process each row in the filtered data
for index, row in filtered_data.iterrows():
    if index % 100 == 0:
        print("Processing index:", index)
        
    image_name = row['image']
    label = row['label']
    
    # Create label folder if it does not exist
    label_folder_path = os.path.join(destination_folder_path, str(label))
    if not os.path.exists(label_folder_path):
        os.makedirs(label_folder_path)
    
    # Read image
    image_path = os.path.join(train_folder_path, image_name)
    image = cv2.imread(image_path)
    
    if image is not None:
        # Resize image to 160x160
        resized_image = cv2.resize(image, (160, 160))
        
        # Save resized image to the label folder
        destination_image_path = os.path.join(label_folder_path, image_name)
        cv2.imwrite(destination_image_path, resized_image)
    else:
        print(f"Warning: Image {image_name} could not be read.")

print("Image processing completed.")
