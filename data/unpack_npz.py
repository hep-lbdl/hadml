import numpy as np
import argparse, os

# Getting the command line argument
parser = argparse.ArgumentParser("numpy_archive_unpacker")
parser.add_argument("--filename", help="Archive name (filename.npz)", required=True)
args = parser.parse_args()

# Unpacking the .npz file
filename = args.filename
print(f"Loading {filename}")
data = np.load(filename)

# Creating a directory for storing NPY files
output_path = os.path.normpath("data/Herwig/cache/")
if not os.path.exists(output_path):
    os.makedirs(output_path)

# Iterate through the arrays and save them as .npy files
for array_name in data.files:
    print(f"Saving {array_name}.npy")
    np.save(os.path.join(output_path, f'{array_name}.npy'), data[array_name])

# Close the .npz file
data.close()