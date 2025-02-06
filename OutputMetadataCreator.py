import os
import csv
import json
from PIL import Image
from DataModels.AnnotatedData import AnnotatedData 
from DataModels.Region import Region

def crop_and_save_regions(image_path: str, regions: list[Region], output_folder: str, base_filename: str):
    """
    Crops regions from an image and saves them to the output folder.
    Returns a list of tuples containing the cropped image path and the corresponding text.
    """
    cropped_data = []
    if not os.path.exists(image_path):
        print(f"Skipping {image_path}: Image file not found.")
        return cropped_data

    img = Image.open(image_path)
    for idx, region in enumerate(regions):
        try:
            # Extract region coordinates
            x, y, width, height = (
                region.shape_attributes.x,
                region.shape_attributes.y,
                region.shape_attributes.width,
                region.shape_attributes.height,
            )
            # Crop the region
            cropped_img = img.crop((x, y, x + width, y + height))
            cropped_img = cropped_img.convert("RGB")

            # Generate the cropped image name
            cropped_image_name = f"{base_filename}_{idx + 1}.jpg"
            cropped_image_path = os.path.join(output_folder, cropped_image_name)

            # Save the cropped image
            cropped_img.save(cropped_image_path)

            # Extract text from region attributes
            text = ""
            if region.region_attributes.medicine_name:
                text = region.region_attributes.medicine_name
            elif region.region_attributes.dosage:
                text = region.region_attributes.dosage
            elif region.region_attributes.dignostic:
                text = region.region_attributes.dignostic
            elif region.region_attributes.symptoms:
                text = region.region_attributes.symptoms
            elif region.region_attributes.personal_info:
                text = region.region_attributes.personal_info
            elif region.region_attributes.numeric_data:
                text = region.region_attributes.numeric_data
            elif region.region_attributes.text:
                text = region.region_attributes.text
            text.replace("\n","").replace("\"","").replace(",","`")
            # Add to the list of cropped data
            cropped_data.append((cropped_image_path, text))
        except Exception as e:
            print(f"Error cropping region {idx + 1} from {image_path}: {e}")

    return cropped_data

def process_folders_to_csv_and_crop(base_folder: str, output_csv: str, cropped_images_folder: str):
    """
    Processes multiple dr folders containing JSON annotations and images.
    Crops regions from images, saves them to a folder, and consolidates into a single CSV file.
    """
    os.makedirs(cropped_images_folder, exist_ok=True)  # Ensure cropped images folder exists

    # Initialize CSV data
    csv_data = [["Cropped Image Path", "Text"]]

    # Loop through all folders starting with 'dr'
    for folder_name in os.listdir(base_folder):
        folder_path = os.path.join(base_folder, folder_name)
        if not os.path.isdir(folder_path) or not folder_name.startswith("dr"):
            continue  # Skip if not a valid dr folder

        json_path = os.path.join(folder_path, f"{folder_name}.json")
        if not os.path.exists(json_path):
            print(f"Skipping {folder_path}: No JSON file found.")
            continue

        with open(json_path, "r", encoding="utf-8") as file:
            data = json.load(file)
            annotated_data = AnnotatedData(data)

        # Process each image in the annotated data
        for image_id, metadata in annotated_data.metadata.items():
            image_path = os.path.join(folder_path, metadata.filename)
            base_filename = os.path.splitext(metadata.filename)[0]  # Remove file extension

            # Crop regions and save to folder
            cropped_data = crop_and_save_regions(image_path, metadata.regions, cropped_images_folder, base_filename)
            # Add cropped data to CSV data
            csv_data.extend(cropped_data)

    # Write to a single CSV file
    with open(output_csv, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerows(csv_data)

    print(f"CSV file created: {output_csv}")

def clean_second_column(overall_output_csv, output_file):
    with open(overall_output_csv, mode='r', newline='', encoding='utf-8') as infile, \
         open(output_file, mode='w', newline='', encoding='utf-8') as outfile:
        
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        
        for row in reader:
            if len(row) > 1:  # Ensure the second column exists
                row[1] = row[1].replace(',', '').replace('"', '').replace('\n', ' ')
            if len(row[1].strip()) > 0:
                writer.writerow(row)
    os.remove(overall_output_csv)

# Usage Example
base_folder = "./base_data"  # Base directory containing dr folders
overall_output_csv = "./all_cropped_data.csv"  # Single output CSV file
overall_output_csv_cleaned = "./all_cropped_data_cleaned.csv"  # Single output CSV file
cropped_images_folder = "./all_cropped_images"  # Folder to save all cropped images

process_folders_to_csv_and_crop(base_folder, overall_output_csv, cropped_images_folder)
clean_second_column(overall_output_csv,overall_output_csv_cleaned)