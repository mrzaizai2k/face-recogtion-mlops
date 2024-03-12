from sklearn.datasets import fetch_lfw_people
import os
from os.path import join, exists
import matplotlib.pyplot as plt

def save_limited_lfw_faces(lfw_path="data/lfw_faces", num_people=500):
    """Saves one face for each of the specified number of unique people.

    Args:
        lfw_path (str, optional): Path to save the LFW faces folder. Defaults to "lfw_faces".
        num_people (int, optional): The number of unique people's faces to download and save. Defaults to 500.
    """
    # Fetch the LFW dataset
    lfw_people = fetch_lfw_people(min_faces_per_person=50, resize=0.5)
    images, targets, target_names = lfw_people.images, lfw_people.target, lfw_people.target_names

    if not exists(lfw_path):
        os.mkdir(lfw_path)

    saved_people = set()  # Track the individuals who have already had a face saved
    num_saved = 0  # Counter for the number of saved faces

    for i, (image, target) in enumerate(zip(images, targets)):
        if num_saved >= num_people:
            break  # Stop once we've saved the desired number of people's faces

        name = target_names[target].replace(" ", "_")  # Create a valid folder/file name

        # Save an image only if this person hasn't been saved yet
        if name not in saved_people:
            person_folder = join(lfw_path, name)

            if not exists(person_folder):
                os.mkdir(person_folder)

            # Save the image with the person's name as the filename
            image_path = join(person_folder, f"{name}.png")

            # Save the image using matplotlib's plt.imsave
            plt.imsave(image_path, image, cmap='gray')

            saved_people.add(name)  # Mark this person as saved
            num_saved += 1  # Increment the saved counter

            if num_saved % 100 == 0 or num_saved == num_people:
                print(f"Saved {num_saved} unique faces...")

    print(f"Total of {num_saved} unique faces saved to {lfw_path}.")

# Run the function to save faces of 500 unique people
save_limited_lfw_faces()
