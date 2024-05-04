import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the image
image = cv2.imread('input_image.jpg')

if image is None:
    print("Error: Unable to load image. Please check the file path.")
    exit()

# Get image dimensions
height, width, _ = image.shape

# Define transformation parameters
scale_factor = 0.5
angle = 45
theta = np.radians(angle)
shear_x = 0.5
shear_y = 0.5

# Construct scaling matrix
scaling_matrix = np.array([[scale_factor, 0, 0],
                            [0, scale_factor, 0],
                            [0, 0, 1]])

# Construct rotation matrix
rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                            [np.sin(theta), np.cos(theta), 0],
                            [0, 0, 1]])

# Construct shearing matrix
shear_matrix = np.array([[1, shear_x, 0],
                         [shear_y, 1, 0],
                         [0, 0, 1]], dtype=np.float32)

# Apply transformations using matrix multiplication
scaled_image = cv2.warpPerspective(image, scaling_matrix, (int(width*scale_factor), int(height*scale_factor)))
rotated_image = cv2.warpPerspective(image, rotation_matrix, (width, height))
sheared_image = cv2.warpPerspective(image, shear_matrix, (width, height))


def show_seperately():
    cv2.imshow('Original Image', image)
    cv2.imshow('Scaled Image', scaled_image)
    cv2.imshow('Rotated Image', rotated_image)
    cv2.imshow('Sheared Image', sheared_image)

    # Wait for any key to be pressed
    cv2.waitKey(0)

    # Close all OpenCV windows
    cv2.destroyAllWindows()


def show_together():
    # Display the original and transformed images
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    # Original image
    axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original Image\n{}x{}'.format(width, height))
    axes[0].axis('off')

    # Scaled image
    axes[1].imshow(cv2.cvtColor(scaled_image, cv2.COLOR_BGR2RGB))
    axes[1].set_title('Scaled Image\n{}x{}'.format(int(width*scale_factor), int(height*scale_factor)))
    axes[1].axis('off')

    # Rotated image
    axes[2].imshow(cv2.cvtColor(rotated_image, cv2.COLOR_BGR2RGB))
    axes[2].set_title('Rotated Image\n{}x{}'.format(width, height))
    axes[2].axis('off')

    # Sheared image
    axes[3].imshow(cv2.cvtColor(sheared_image, cv2.COLOR_BGR2RGB))
    axes[3].set_title('Sheared Image\n{}x{}'.format(width, height))
    axes[3].axis('off')

    plt.tight_layout()
    plt.show()

# show_seperately()
show_together()

