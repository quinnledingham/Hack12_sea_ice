import os
import numpy as np
from osgeo import gdal, osr
import glob
from tqdm import tqdm
from collections import Counter


def extract_patches_with_overlap(gt_path, hh_path, hv_path, output_dir, patch_size=256, overlap_pixels=20):
    """
    Extract patches from RCM HH and HV images based on ground truth classes 2,3,4
    with 20 pixels overlap and preserve geospatial information.
    Files are named after the dominant class ID.
    """

    # Create output directories
    os.makedirs(os.path.join(output_dir, 'gt'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'hh'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'hv'), exist_ok=True)

    # Open datasets
    gt_ds = gdal.Open(gt_path)
    hh_ds = gdal.Open(hh_path)
    hv_ds = gdal.Open(hv_path)

    # Get raster properties
    gt_transform = gt_ds.GetGeoTransform()
    projection = gt_ds.GetProjection()
    x_size = gt_ds.RasterXSize
    y_size = gt_ds.RasterYSize

    # Calculate step size with overlap
    step = patch_size - overlap_pixels

    # Counter for saved patches
    patch_count = 0

    # Iterate through the image with overlap
    for y in range(0, y_size - patch_size + 1, step):
        for x in range(0, x_size - patch_size + 1, step):

            # Read ground truth patch
            gt_patch = gt_ds.ReadAsArray(x, y, patch_size, patch_size)

            # Check if patch contains any of classes 2,3,4
            if np.any((gt_patch >= 2) & (gt_patch <= 4)):
                # Check if patch is NOT 100% class 0 or 1
                if not (np.all((gt_patch == 0) | (gt_patch == 1))):

                    # Read corresponding HH and HV patches
                    hh_patch = hh_ds.ReadAsArray(x, y, patch_size, patch_size)
                    hv_patch = hv_ds.ReadAsArray(x, y, patch_size, patch_size)

                    # Calculate new geotransform for the patch
                    new_gt = (
                        gt_transform[0] + x * gt_transform[1],  # upper left x
                        gt_transform[1],  # pixel width
                        gt_transform[2],  # rotation
                        gt_transform[3] + y * gt_transform[5],  # upper left y
                        gt_transform[4],  # rotation
                        gt_transform[5]  # pixel height
                    )

                    # Get dominant class (excluding 0 and 1)
                    valid_pixels = gt_patch[(gt_patch >= 2) & (gt_patch <= 4)]
                    if len(valid_pixels) > 0:
                        class_counter = Counter(valid_pixels)
                        dominant_class = class_counter.most_common(1)[0][0]
                    else:
                        dominant_class = 2  # fallback to first year ice

                    # Create filename with class ID and coordinates
                    ulx = new_gt[0]
                    uly = new_gt[3]
                    filename = f"class_{dominant_class}_x{int(ulx)}_y{int(uly)}.tif"

                    # Save patches as GeoTIFF
                    save_geotiff(os.path.join(output_dir, 'gt', filename),
                                 gt_patch, new_gt, projection, gdal.GDT_Byte)

                    save_geotiff(os.path.join(output_dir, 'hh', filename),
                                 hh_patch, new_gt, projection, hh_ds.GetRasterBand(1).DataType)

                    save_geotiff(os.path.join(output_dir, 'hv', filename),
                                 hv_patch, new_gt, projection, hv_ds.GetRasterBand(1).DataType)

                    patch_count += 1

    # Close datasets
    gt_ds = None
    hh_ds = None
    hv_ds = None

    print(f"Extracted {patch_count} patches for {os.path.basename(gt_path)}")


def save_geotiff(output_path, array, geotransform, projection, data_type):
    """Save numpy array as GeoTIFF with proper geospatial information."""

    driver = gdal.GetDriverByName('GTiff')
    rows, cols = array.shape

    # Create output dataset
    out_ds = driver.Create(output_path, cols, rows, 1, data_type)

    # Set geotransform and projection
    out_ds.SetGeoTransform(geotransform)
    out_ds.SetProjection(projection)

    # Write array to band
    out_band = out_ds.GetRasterBand(1)
    out_band.WriteArray(array)

    # Close dataset
    out_ds = None


def extract_patches_by_class_distribution(gt_path, hh_path, hv_path, output_dir, patch_size=256, overlap_pixels=20):
    """
    Alternative version that creates separate folders for each class
    and names files with coordinates only.
    """

    # Create output directories for each class
    for class_id in [2, 3, 4]:
        os.makedirs(os.path.join(output_dir, f'class_{class_id}', 'gt'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, f'class_{class_id}', 'hh'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, f'class_{class_id}', 'hv'), exist_ok=True)

    # Open datasets
    gt_ds = gdal.Open(gt_path)
    hh_ds = gdal.Open(hh_path)
    hv_ds = gdal.Open(hv_path)

    # Get raster properties
    gt_transform = gt_ds.GetGeoTransform()
    projection = gt_ds.GetProjection()
    x_size = gt_ds.RasterXSize
    y_size = gt_ds.RasterYSize

    # Calculate step size with overlap
    step = patch_size - overlap_pixels

    # Counter for saved patches
    patch_count = 0

    # Iterate through the image with overlap
    for y in range(0, y_size - patch_size + 1, step):
        for x in range(0, x_size - patch_size + 1, step):

            # Read ground truth patch
            gt_patch = gt_ds.ReadAsArray(x, y, patch_size, patch_size)

            # Check if patch contains any of classes 2,3,4
            if np.any((gt_patch >= 2) & (gt_patch <= 4)):
                # Check if patch is NOT 100% class 0 or 1
                if not (np.all((gt_patch == 0) | (gt_patch == 1))):

                    # Read corresponding HH and HV patches
                    hh_patch = hh_ds.ReadAsArray(x, y, patch_size, patch_size)
                    hv_patch = hv_ds.ReadAsArray(x, y, patch_size, patch_size)

                    # Calculate new geotransform for the patch
                    new_gt = (
                        gt_transform[0] + x * gt_transform[1],  # upper left x
                        gt_transform[1],  # pixel width
                        gt_transform[2],  # rotation
                        gt_transform[3] + y * gt_transform[5],  # upper left y
                        gt_transform[4],  # rotation
                        gt_transform[5]  # pixel height
                    )

                    # Get class distribution
                    for class_id in [2, 3, 4]:
                        class_pixels = np.sum(gt_patch == class_id)
                        if class_pixels > 0:  # If this class exists in the patch

                            # Create filename with coordinates
                            ulx = new_gt[0]
                            uly = new_gt[3]
                            filename = f"x{int(ulx)}_y{int(uly)}.tif"

                            # Save patches as GeoTIFF in class-specific folder
                            save_geotiff(os.path.join(output_dir, f'class_{class_id}', 'gt', filename),
                                         gt_patch, new_gt, projection, gdal.GDT_Byte)

                            save_geotiff(os.path.join(output_dir, f'class_{class_id}', 'hh', filename),
                                         hh_patch, new_gt, projection, hh_ds.GetRasterBand(1).DataType)

                            save_geotiff(os.path.join(output_dir, f'class_{class_id}', 'hv', filename),
                                         hv_patch, new_gt, projection, hv_ds.GetRasterBand(1).DataType)

                    patch_count += 1

    # Close datasets
    gt_ds = None
    hh_ds = None
    hv_ds = None

    print(f"Extracted {patch_count} patches for {os.path.basename(gt_path)}")


def main():
    # Configuration
    base_dir = '/beluga/Hack12_multi_type_seaice'  # Change this to your data directory
    patch_size = 256
    overlap_pixels = 20  # 20 pixels overlap

    # File paths - adjust these according to your actual file names
    hh_image_path = '/home/yimin/2025/Hackathon_Sea_Ice_Typing/sar_images/RCM3-OK3472315-PK3534862-1-SCLNA-20250322-124532-HH-HV-GRD_hh-resolution-200m-reprojected-merged-resized.tiff'  # Replace with your HH image path
    hv_image_path = '/home/yimin/2025/Hackathon_Sea_Ice_Typing/sar_images/RCM3-OK3472315-PK3534862-1-SCLNA-20250322-124532-HH-HV-GRD_hv-resolution-200m-reprojected-merged-resized.tiff'  # Replace with your HV image path

    # Ground truth files
    gt_files = {
        'train': '/home/yimin/2025/Hackathon_Sea_Ice_Typing/gt_tiff/train_gt.tif',
        'val': '/home/yimin/2025/Hackathon_Sea_Ice_Typing/gt_tiff/val_gt.tif',
        'test': '/home/yimin/2025/Hackathon_Sea_Ice_Typing/gt_tiff/test_gt.tif'
    }

    # Process each dataset
    for dataset_name, gt_file in gt_files.items():
        print(f"Processing {dataset_name} dataset...")

        gt_path = os.path.join(base_dir, gt_file)
        output_dir = os.path.join(base_dir, f'{dataset_name}_patches')

        # Check if files exist
        if not all(os.path.exists(path) for path in [gt_path, hh_image_path, hv_image_path]):
            print(f"Warning: Some files missing for {dataset_name}. Skipping...")
            continue

        # Extract patches (choose one method)
        print("Using method 1: Files named with class ID and coordinates")
        extract_patches_with_overlap(
            gt_path,
            hh_image_path,
            hv_image_path,
            output_dir,
            patch_size,
            overlap_pixels
        )

        # Uncomment below if you want the alternative method with class folders
        # print("Using method 2: Separate folders for each class")
        # extract_patches_by_class_distribution(
        #     gt_path,
        #     hh_image_path,
        #     hv_image_path,
        #     output_dir + "_by_class",
        #     patch_size,
        #     overlap_pixels
        # )


if __name__ == "__main__":
    main()