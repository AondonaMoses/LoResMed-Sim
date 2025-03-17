import torch, os
import torchio as tio
import nibabel as nib
import numpy as np
from pathlib import Path
import json

# Custom Transforms
class RandomRicianNoise(tio.RandomNoise):
    """Apply Rician noise instead of Gaussian noise in TorchIO's RandomNoise."""

    def get_noise(self, tensor: torch.Tensor, mean: float, std: float) -> torch.Tensor:
        """Generate Rician noise instead of Gaussian noise."""
        noise_real = torch.randn_like(tensor) * std + mean
        noise_imag = torch.randn_like(tensor) * std + mean
        return torch.sqrt((tensor + noise_real) ** 2 + noise_imag ** 2)

class RandomCrop(tio.Transform):
    def __init__(self, crop_percentage_range=(0.2, 0.5), min_fov_percentage=0.5, p=0.1, **kwargs):
        super().__init__(**kwargs)
        self.crop_percentage_range = crop_percentage_range
        self.min_fov_percentage = min_fov_percentage
        self.p = p
        self.args_names = ('crop_percentage_range', 'min_fov_percentage')  # For torchio history

    def apply_transform(self, subject):
        if np.random.rand() < self.p:
            d, h, w = subject.get_first_image().shape[1:]
            min_dim = min(d, h, w)
            max_crop_percentage = 1.0 - self.min_fov_percentage
            crop_percentage = np.random.uniform(max(self.crop_percentage_range[0], 0.0),
                                            min(self.crop_percentage_range[1], max_crop_percentage))
            crop_total = int(min_dim * crop_percentage)
            crop_total = crop_total if crop_total % 2 == 0 else crop_total + 1
            d_front = d_back = min(crop_total // 2, d // 2)
            h_front = h_back = min(crop_total // 2, h // 2)
            w_front = w_back = min(crop_total // 2, w // 2)
            cropping = (d_front, d_back, h_front, h_back, w_front, w_back)
            
            return tio.Crop(cropping)(subject)
        return subject
            

class RandomPatchDegradation(tio.Transform):
    def __init__(self, num_patches=5, intensity_range=(0.7, 0.9), **kwargs):
        super().__init__(**kwargs)
        self.num_patches, self.intensity_range = num_patches, intensity_range

    def apply_transform(self, subject):
        data = subject.get_first_image().data.clone().to(torch.float32)
        d, h, w = data.shape[1:]
        for _ in range(self.num_patches):
            d_start, h_start, w_start = [np.random.randint(0, s // 2) for s in (d, h, w)]
            patch_size = np.random.randint(10, 30)
            intensity = np.random.uniform(*self.intensity_range)
            d_end = min(d_start + patch_size, d)
            h_end = min(h_start + patch_size, h)
            w_end = min(w_start + patch_size, w)
            data[:, d_start:d_end, h_start:h_end, w_start:w_end] += intensity
        data = torch.clamp(data, 0, 1)
        subject.get_first_image().data = data
        return subject


class SparseSpatialTransform(tio.Transform):
    def __init__(self, p_apply=0.75, max_effects=3, weights=(0.3, 0.5, 0.2), **kwargs):
        super().__init__(**kwargs)
        self.p_apply, self.max_effects = p_apply, min(max_effects, 3)
        self.weights = weights / np.sum(weights)
        self.effects = [
            tio.RandomNoise(mean=0, std=(lambda: np.random.uniform(0.05, 0.15))()),  # RandomNoise -0.2, RandomAnisotropy -0.5 RandomBiasField - 0.2
            tio.RandomAnisotropy(axes=(0, 1, 2), downsampling=(2, 6)),
            tio.RandomBiasField(coefficients=(lambda: np.random.uniform(0.3, 0.7))())
        ]

    def apply_transform(self, subject):
        if np.random.rand() < self.p_apply:
            num_effects = np.random.choice([1, 2, 3], p=self.weights[:self.max_effects])
            chosen_indices = np.random.choice([0, 1, 2], size=num_effects, replace=False)
            for idx in chosen_indices:
                subject = self.effects[idx](subject)
        #SparseSpatialTransform(p_apply=0.6, max_effects=3, weights=(0.2, 0.5, 0.3)),  ## RandomNoise -0.2, RandomAnisotropy -0.5 RandomBiasField - 0.2
        return subject

class RandomResample(tio.Transform):
    def apply_transform(self, subject):
        voxel_size = np.random.choice([2, 3])
        return tio.Resample(voxel_size)(subject)
    
## Explicitly register custom tio transforms
tio.transforms.SparseSpatialTransform = SparseSpatialTransform
tio.transforms.__dict__["SparseSpatialTransform"] = SparseSpatialTransform
tio.transforms.RandomRicianNoise = RandomRicianNoise
tio.transforms.__dict__["RandomRicianNoise"] = RandomRicianNoise
tio.transforms.RandomCrop = RandomCrop
tio.transforms.__dict__["RandomCrop"] = RandomCrop



# Will try this out on segmentation task
tio.transforms.RandomPatchDegradation = RandomPatchDegradation
tio.transforms.__dict__["RandomPatchDegradation"] = RandomPatchDegradation

# Core Functions
def get_degradation_pipeline():
    return tio.Compose([
        RandomResample(),
        tio.RandomBiasField(coefficients=(lambda: np.random.uniform(0.3, 0.7))(), p=0.3),
        tio.OneOf({                            # Noise Artifacts
            RandomRicianNoise(std=(0.05, 0.15)): 2,
            tio.RandomNoise(mean=0, std=(lambda: np.random.uniform(0.05, 0.15))()): 1
        }, p=0.2),
        tio.OneOf({
            tio.RandomMotion(degrees=(lambda: np.random.uniform(3, 7))(), translation=(lambda: np.random.uniform(3, 7))(), num_transforms=2): 2,
            tio.RandomBlur(std=(lambda: np.random.uniform(0.3, 0.7))(), p=0.3): 1,
        }, p=0.1),
        tio.RandomSpike(num_spikes=(lambda: np.random.randint(5, 15))(), intensity=(lambda: np.random.uniform(0.1, 0.3))(), p=0.05),
        tio.RandomGhosting(intensity=(lambda: np.random.uniform(0.1, 0.3))(), axes=(lambda: np.random.randint(0, 3))(), p=0.01),
        RandomCrop(p=0.1)  # Uniform crop for limited FOV
    ])
    #RandomPatchDegradation(num_patches=5, intensity_range=(0.1, 0.3)): #Use for segmentation masks


def get_individual_transforms():
    return {
        "Resample": RandomResample(),
        "SparseSpatial": SparseSpatialTransform(p_apply=0.6, max_effects=3, weights=(0.2, 0.5, 0.3)),
        "Rician Noise": RandomRicianNoise(std=(0.05, 0.15)),
        "Noise": tio.RandomNoise(mean=0, std=(lambda: np.random.uniform(0.05, 0.15))()),
        "Anisotropy": tio.RandomAnisotropy(axes=(0, 1, 2), downsampling=(2, 5)),
        "BiasField": tio.RandomBiasField(coefficients=(lambda: np.random.uniform(0.3, 0.7))()),
        "Blur": tio.RandomBlur(std=(lambda: np.random.uniform(0.3, 0.7))()),
        "Motion": tio.RandomMotion(degrees=(lambda: np.random.uniform(3, 7))(), translation=(lambda: np.random.uniform(3, 7))(), num_transforms=2),
        "Spike": tio.RandomSpike(num_spikes=(lambda: np.random.randint(5, 15))(), intensity=(lambda: np.random.uniform(0.1, 0.3))()),
        "Ghosting": tio.RandomGhosting(intensity=(lambda: np.random.uniform(0.1, 0.3))(), axes=(lambda: np.random.randint(0, 3))()),
        "RandomCrop": RandomCrop(p=1),  # Uniform crop for limited FOV
        #"PatchDegradation": RandomPatchDegradation(num_patches=5, intensity_range=(0.1, 0.3)) #Use for segmentation masks
    }

def save_transform_history(subjects, output_json='transform_history.json'):
    history = subjects.get_composed_history()
    print(history)
    with open(output_json, 'w') as f:
        json.dump(history, f, indent=4)
    return output_json

def save_history_to_jsons(history_dict, output_json='transform_history.json'):
    """Saves the transformation history dictionary to a JSON file."""
    with open(output_json, 'w') as f:
        json.dump(history_dict, f, indent=4, default=str) #use default=str to handle non-serializable objects.
        return output_json


def save_history_to_json(history_dict, output_json='transform_history.json'):
    """Updates the transformation history JSON file with new key-value pairs without overwriting the entire dictionary."""
    
    # Load existing data if the file exists
    if os.path.exists(output_json):
        with open(output_json, 'r') as f:
            try:
                existing_data = json.load(f)
                if not isinstance(existing_data, dict):  # Ensure it's a dictionary
                    raise ValueError("Existing JSON file is not a dictionary!")
            except (json.JSONDecodeError, ValueError):
                existing_data = {}  # Default to empty dictionary if file is corrupted or not a dictionary
    else:
        existing_data = {}

    # Merge new history entries into existing dictionary
    existing_data.update(history_dict)

    # Write back to JSON file
    with open(output_json, 'w') as f:
        json.dump(existing_data, f, indent=4, default=str)  # Pretty print for readability
    
    return output_json




def degrade_mri(input_path, output_path, pipeline=None):
    #print('output_path:', output_path)
    
    Path('outputs/').mkdir(exist_ok=True)
    
    if pipeline is None:
        img_basename = output_path.split('/')[-1].replace('.nii.gz', '')
    else:
        img_basename = os.path.basename(input_path)
        
    
    try:
        nifti_img = nib.load(input_path)
        image_data = nifti_img.get_fdata()
        affine = nifti_img.affine
        image_data = torch.from_numpy(image_data).float().unsqueeze(0)
        subject = tio.Subject(
            image=tio.ScalarImage(tensor=image_data, 
                                  affine=affine, 
                                  type=tio.INTENSITY)) #Include affine
        
    except FileNotFoundError:
        print(f"Error: File not found at {input_path}. Please ensure the path is correct.")
        exit()  # Exit if file is not found.
    except Exception as e:
        print(f"Error loading NIfTI file: {e}")
        exit()
        
    pipeline = pipeline or get_degradation_pipeline()
    degraded_mri = pipeline(subject)
    
    history_dict = {
    img_basename: [str(t) for t in degraded_mri.history]
    }

    history_path = output_path.replace('.nii.gz', '_history.json')
    save_history_to_json(history_dict, os.path.join('outputs', '_history.json'))
    
    if pipeline is not None:
        degraded_mri.image.save(os.path.join('outputs', output_path))
    else:
        degraded_mri.save(os.path.join('outputs', img_basename))
    
    return degraded_mri, history_path