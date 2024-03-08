import os

def rename_and_sort_images(folder_path):
    # List all files in the folder
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    # Sort files
    files.sort()

    # Set the base name for renaming
    base_name = "image_"

    # Iterate through files and rename
    for index, file_name in enumerate(files, start=1):
        # Extract file extension
        _, extension = os.path.splitext(file_name)

        # Create the new file name
        new_name = f"{base_name}{index}_B{extension}"

        # Build the full paths
        old_path = os.path.join(folder_path, file_name)
        new_path = os.path.join(folder_path, new_name)

        # Rename the file
        os.rename(old_path, new_path)
        print(f'Renamed: {old_path}  ->  {new_path}')
        files.sort()


# Specify the path to the folder containing your images
folder_path = 'output'

# Call the function to rename and sort images
rename_and_sort_images(folder_path)
