import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load the NIfTI file
nii_file = nib.load('path_to_your_file.nii')  # replace with your .nii file path
nii_data = nii_file.get_fdata()

# Step 2: Define function to normalize slice to [0, 255] range
def normalize_slice(slice_data):
    slice_normalized = (slice_data - np.min(slice_data)) / (np.max(slice_data) - np.min(slice_data)) * 255
    return slice_normalized.astype(np.uint8)

# Step 3: Select middle slices for each view
slice_index_axial = nii_data.shape[2] // 2    # Middle slice along Z-axis for Axial view
slice_index_coronal = nii_data.shape[1] // 2  # Middle slice along Y-axis for Coronal view
slice_index_sagittal = nii_data.shape[0] // 2 # Middle slice along X-axis for Sagittal view

# Step 4: Extract slices
axial_slice = nii_data[:, :, slice_index_axial]    # Axial (Z-axis slice)
coronal_slice = nii_data[:, slice_index_coronal, :] # Coronal (Y-axis slice)
sagittal_slice = nii_data[slice_index_sagittal, :, :] # Sagittal (X-axis slice)

# Step 5: Normalize slices
axial_slice_normalized = normalize_slice(axial_slice)
coronal_slice_normalized = normalize_slice(coronal_slice)
sagittal_slice_normalized = normalize_slice(sagittal_slice)

# Step 6: Save the slices as images
plt.imsave('axial_view.png', axial_slice_normalized, cmap='gray')
plt.imsave('coronal_view.png', coronal_slice_normalized, cmap='gray')
plt.imsave('sagittal_view.png', sagittal_slice_normalized, cmap='gray')

print("Images saved successfully!")
