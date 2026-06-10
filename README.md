# Brain Tumor Segmentation

Brain-tumor segmentation on MRI volumes (BraTS dataset, NIfTI `.nii` format)
combining classical **digital image processing** with a **YOLO** oriented
bounding-box / segmentation model, plus a metrics suite to evaluate the results.

The pipeline slices 3D MRI volumes (axial / coronal planes), enhances them
(CLAHE, frequency- and spatial-domain filtering), localizes the tumor region and
produces a segmentation mask, then scores the masks against ground truth.

## Project layout

| Path | Description |
|------|-------------|
| `main.py` | Main segmentation pipeline over `.nii` MRI volumes. |
| `metrics.py` | Computes segmentation metrics (Jaccard/IoU, Panoptic Quality). |
| `validation.py` | Validation routines for the segmentation output. |
| `utils/` | Image-processing helpers: `digital_image_processing`, `frequency_domain`, `spartial_domain`. |
| `images/` | Reference masks and templates used during processing. |
| `docs/` | Supporting documentation. |
| `*.csv` | Computed metrics and segmentation indices. |

## Requirements

- Python 3.9+ (CUDA-capable GPU recommended for YOLO)
- `ultralytics`, `nibabel`, `opencv-python`, `numpy`, `pandas`, `matplotlib`,
  `torch`, `torchmetrics`

## Notes

The scripts reference absolute local paths for the dataset, the trained YOLO
weights (`best.pt`) and results directories — update these to your own
environment before running. Datasets and trained weights are not part of this
repository.
