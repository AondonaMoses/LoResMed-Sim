# LoResMed-Sim: Low-Resolution Medical Imaging Simulator
LoResMed-Sim Demo [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1q49_Y9pMlXLi9y614JV3ilYjXBTr7Rqc?usp=sharing)

This pipeline leverages `torchio` to degrade high-quality MRI scans, replicating the imaging artifacts and limitations prevalent in `resource-constrained settings`. It simulates the effects of `outdated hardware`, `suboptimal calibration`, and `operational constraints`, enabling robust testing/training of image analysis algorithms under realistic conditions. Below are the implemented degradation transforms:

## Degradation Transforms

### Low Resolution
To emulate the reduced spatial resolution of legacy LMIC scanners:
- **Resample**: `tio.Resample()` randomly adjusts voxel sizes to `[2, 3, 4, 5, 6]` mm, reflecting coarse sampling.
- **Anisotropy**: `tio.RandomAnisotropy()` downsamples a single axis (`[0, 1, 2]`) by a factor of 2–6, producing anisotropic voxels (e.g., 1x1x5mm), typical of rapid acquisitions.
- **Universal Application**: One of these is applied to all images to ensure consistent resolution degradation.

### Magnetic Field Inhomogeneity (MFI)
To replicate contrast distortions from poorly shimmed magnets:
- **Bias Field**: `tio.RandomBiasField` introduces intensity variations (coefficients 0.3–0.7) with a 30% probability, simulating field non-uniformity common in aging systems.

### Noise Artifacts
To model signal corruption from inadequate shielding or power instability:
- **Gaussian Noise**: `tio.RandomNoise` applies Gaussian noise (std 0.05–0.15).
- **Rician Noise**: `RandomRicianNoise` (std 0.05–0.15), weighted 2:1, reflects MRI-specific noise profiles.
- **Probability**: Applied to 20% of images, balancing realism and frequency.

### Motion Artifacts
To simulate patient motion during prolonged scan times:
- **Motion**: `tio.RandomMotion` (degrees & translation 3–7, weighted 2:1) introduces displacement artifacts.
- **Blur**: `tio.RandomBlur` (std 0.3–0.7, 30% internal probability) mimics motion-induced softening.
- **Occurrence**: Affects 10% of images, consistent with clinical motion prevalence.

### Compression Artifacts
To approximate data loss from aggressive compression:
- **Spikes**: `tio.RandomSpike` (5–15 spikes, intensity 0.1–0.3) introduces block-like distortions, applied to 5% of images.

### MRI Ghosting
To emulate aliasing from scanner instability or physiological motion:
- **Ghosting**: `tio.RandomGhosting` (intensity 0.1–0.3, random axis `[0, 1, 2]`) affects 1% of images, reflecting rare but impactful artifacts.

### Limited Field of View (FOV)
To simulate incomplete anatomical coverage:
- **Crop**: `RandomCrop` uniformly reduces each axis by 20–50% (retaining 50% anatomy), applied to 10% of images, mimicking hardware or positioning limitations.

## Implementation

The degradation pipeline is implemented as a `tio.Compose` sequence:

```python
tio.Compose([
    RandomResample(),  # Resolution degradation
    tio.RandomBiasField(coefficients=(lambda: np.random.uniform(0.3, 0.7))(), p=0.3),  # MFI
    tio.OneOf({RandomRicianNoise(std=(0.05, 0.15)): 2, tio.RandomNoise(mean=0, std=(lambda: np.random.uniform(0.05, 0.15))()): 1}, p=0.2),  # Noise
    tio.OneOf({tio.RandomMotion(degrees=(lambda: np.random.uniform(3, 7))(), translation=(lambda: np.random.uniform(3, 7))(), num_transforms=2): 2, tio.RandomBlur(std=(lambda: np.random.uniform(0.3, 0.7))(), p=0.3): 1}, p=0.1),  # Motion
    tio.RandomSpike(num_spikes=(lambda: np.random.randint(5, 15))(), intensity=(lambda: np.random.uniform(0.1, 0.3))(), p=0.05),  # Compression
    tio.RandomGhosting(intensity=(lambda: np.random.uniform(0.1, 0.3))(), axes=(lambda: np.random.randint(0, 3))(), p=0.01),  # Ghosting
    RandomCrop(p=0.1)  # Limited FOV
])
```

## Setup
```bash
git https://github.com/AondonaMoses/LoResMed-Sim.git
cd LoResMed-Sim
pip install -r requirements.txt
```


## Example Usage from Terminal

The simulator provides a command-line interface for degrading MRI datasets, supporting both single-image and batch processing workflows. Each operation produces degraded images alongside transform history logs, ensuring traceability and reproducibility for clinical and research applications.

### Single Image Degradation
- **Description**: Degrades an individual MRI scan using the full degradation pipeline, simulating low-resource imaging conditions. Reads `input.nii.gz`, applies the degradation transforms, and writes the output to `degraded.nii`. A detailed transform history is saved as `degraded_history.json`.
- **Command**:
  ```bash
  python LMICsQ-Sim.py -i sample_data/high_quality_mri.nii.gz -o output/degraded.nii --single
  ```

#### Output Files
- `outputs/degraded.nii.gz`: The degraded MRI scan.
- `outputs/_history.json`: JSON log detailing the applied transforms.

### Batch Image Degradation
- **Description**: Processes multiple MRI scans within a directory, applying the degradation pipeline to each for efficient dataset preparation. Identifies all `.nii` or `.nii.gz` files in `input_dir`, applies the degradation pipeline to each, and saves results in output_dir (e.g., `degraded_file1.nii.gz`). A consolidated transform history is written to `batch_transform_history.json`.
- **Command**:
  ```bash
  python LMICsQ-Sim.py -i sample_data -o output_batch --batch  
  ```
#### Input/Output: 
- `outputs/degraded_mri1.nii.gz`: Degraded version of `sample_data/mri1.nii.gz`.
- `outputs/degraded_mri2.nii.gz`: Degraded version of `sample_data/mri2.nii.gz`.
- `outputs/batch_transform_history.json`: Aggregated transform history for all processed images in JSON format.



## Sample Results
This section showcases an example of the degradation process, comparing an original MRI scan with its degraded counterpart and providing the associated transform history.

**Original Image**
- **File:** `Sample_Data/MNI152_T1_2mm_brain.nii.gz`
- **Visualization:** 
_High-resolution MRI scan prior to degradation._   
![Original MRI](https://github.com/AondonaMoses/LoResMed-Sim/blob/main/Sample_Data/MNI152_T1_2mm_brain.nii.gz)


**Degraded Image**
- **File:** `outputs/full_degraded.nii`
- **Visualization:** 
_Degraded MRI scan exhibiting low-resource imaging artifacts._

![Degraded MRI](https://github.com/AondonaMoses/LoResMed-Sim/blob/main/outputs/full_degraded.nii)


**Transform History**
- **File:** `output/_history.json`
- **Sample Content:** 
  ```json
  {
    "MNI152_T1_2mm_brain.nii.gz": [
        "Resample(target=3, image_interpolation=linear, pre_affine_name=None, scalars_only=False)",
        "RandomResample()",
    ]
  }
  ```
- **Description**: The JSON log details the sequence of transforms applied, for this instance a voxel resampling to 3mm was applied.

## Sample Results
This project utilizes `torchio`, an open-source Python library for medical image processing. For more information and additional examples, refer to:
- **Citation:** Pérez-García, F., et al. "TorchIO: a Python library for efficient loading, preprocessing, augmentation and patch-based sampling of medical images in deep learning." _Computer Methods and Programs in Biomedicine, 2021_.
- **GitHub Repository:** [https://github.com/fepegar/torchio](https://github.com/fepegar/torchio)
- **Notebook:** Explore the [TorchIO Jupyter Notebook](https://github.com/TorchIO-project/torchio/blob/main/tutorials/README.md) for comprehensive examples of image transformations and usage. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TorchIO-project/torchio-notebooks/blob/main/notebooks/Data_preprocessing_and_augmentation_using_TorchIO_a_tutorial.ipynb)





