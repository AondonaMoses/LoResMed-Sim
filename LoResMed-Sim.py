import argparse
import os
import torchio as tio
from degrade_mri import get_degradation_pipeline, save_history_to_json, degrade_mri
import glob
from tqdm import tqdm 

def degrade_single(input_path, output_path):
    """Degrade a single MRI and save with transform history."""
    # Load image
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file {input_path} not found")
    mri = tio.ScalarImage(input_path)
    
    # Apply pipeline
    pipeline = get_degradation_pipeline()
    degraded_mri = pipeline(mri)
    
    # Save output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    degraded_mri.save(output_path)
    
    # Save transform history
    history_path = output_path.replace('.nii', '_history.json')
    save_history_to_json([degraded_mri], history_path)
    print(f"Degraded MRI saved to {output_path}")
    print(f"Transform history saved to {history_path}")

def degrade_batch(input_dir, output_dir):
    
    input_files = glob.glob(os.path.join(input_dir, '*.nii')) + glob.glob(os.path.join(input_dir, '*.nii.gz'))
    if not input_files:
        raise FileNotFoundError(f"No NIfTI files found in {input_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    pipeline = get_degradation_pipeline()
    for input_path in  tqdm(input_files, desc="Degrading MRI files"):
        filename = os.path.basename(input_path)
        output_path = os.path.join(output_dir, f"degraded_{filename}")
        degraded_mri, _ = degrade_mri(input_path, output_path, pipeline)
        print(f"Degraded MRI saved to {output_path}")
        
    print(f"Finished processing {len(input_files)} MRI files.")
    
    # history_path = os.path.join(output_dir, 'batch_transform_history.json')
    # save_history_to_json(degraded_mris, history_path)
    # print(f"Batch transform history saved to {history_path}")

def main():
    parser = argparse.ArgumentParser(description="Degrade MRI scans to simulate low-quality LMIC images.")
    parser.add_argument('-i', '--input', required=True, help="Input NIfTI file or directory")
    parser.add_argument('-o', '--output', type=str, default="/outputs/", required=True, help="Output file path (single) or directory (batch)")
    parser.add_argument('--batch', action='store_true', help="Process as batch")
    
    args = parser.parse_args()
    
    if args.batch:
        degrade_batch(args.input, args.output)
    else:
        degrade_mri(args.input, args.output)
        print(f"Degraded MRI saved to {args.output}")
        print(f"Transform history saved to {args.output.replace('.nii', '_history.json')}")

if __name__ == "__main__":
    main()