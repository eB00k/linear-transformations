import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# Read the image
image = cv2.imread('input_image.jpg')

if image is None:
    print("Error: Unable to load image. Please check the file path.")
    exit()

height, width, _ = image.shape

# Create a figure and axis
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.25)

# Display the image
im = ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

# Define transformation function
def transform_image(val):
    scale_factor = scale_slider.val
    angle = angle_slider.val
    shear_x = shear_x_slider.val
    shear_y = shear_y_slider.val
    
    theta = np.radians(angle)
    scaling_matrix = np.array([[scale_factor, 0, 0],
                               [0, scale_factor, 0],
                               [0, 0, 1]])
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                [np.sin(theta), np.cos(theta), 0],
                                [0, 0, 1]])
    shear_matrix = np.array([[1, shear_x, 0],
                             [shear_y, 1, 0],
                             [0, 0, 1]], dtype=np.float32)
    
    # Apply transformations using matrix multiplication
    transformed_image = cv2.warpPerspective(image, scaling_matrix, (int(width*scale_factor), int(height*scale_factor)))
    transformed_image = cv2.warpPerspective(transformed_image, rotation_matrix, (width, height))
    transformed_image = cv2.warpPerspective(transformed_image, shear_matrix, (width, height))
    
    # Update the displayed image
    im.set_data(cv2.cvtColor(transformed_image, cv2.COLOR_BGR2RGB))
    fig.canvas.draw_idle()

# Add sliders for each transformation parameter
scale_slider_ax = plt.axes([0.25, 0.1, 0.65, 0.03])
scale_slider = Slider(scale_slider_ax, 'Scale Factor', 0.1, 2.0, valinit=0.5)

angle_slider_ax = plt.axes([0.25, 0.05, 0.65, 0.03])
angle_slider = Slider(angle_slider_ax, 'Rotation Angle', -180, 180, valinit=45)

shear_x_slider_ax = plt.axes([0.25, 0.15, 0.65, 0.03])
shear_x_slider = Slider(shear_x_slider_ax, 'Shear X', -1.0, 1.0, valinit=0.0)

shear_y_slider_ax = plt.axes([0.25, 0.2, 0.65, 0.03])
shear_y_slider = Slider(shear_y_slider_ax, 'Shear Y', -1.0, 1.0, valinit=0.0)

# Register the update function with the sliders
scale_slider.on_changed(transform_image)
angle_slider.on_changed(transform_image)
shear_x_slider.on_changed(transform_image)
shear_y_slider.on_changed(transform_image)

plt.show()