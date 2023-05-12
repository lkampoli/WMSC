import os
import argparse
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("file_path")
parser.add_argument("-out", "--out_folder", default="./")
parser.add_argument("-coords", "--get_coords", action="store_true")
parser.add_argument("-dim", "--var_dim", type=int, default=1)
args = parser.parse_args()

file_path = args.file_path
file_name = os.path.basename(file_path)

out_folder = args.out_folder
out_ext = ".edf"

get_coords = args.get_coords
coord_vars = [
    ("X", 1),
    ("Y", 2),
    ("Z", 3),
]

var_dim = args.var_dim

num_vars = (int(subprocess.run(f"head -n 1 {file_path} | wc -w", shell=True, stdout=subprocess.PIPE).stdout.decode().strip()) - len(coord_vars)) // var_dim
print(f"{num_vars} variables detected:")

file_vars = file_name.split('.')[0].split('_')[-num_vars:]
print(', '.join(file_vars))

for v, var in enumerate(file_vars):
    print(f"Extract {var}.")

    scol = v*var_dim + len(coord_vars) + 1
    ecol = scol + var_dim - 1
    out_path = os.path.join(out_folder, var.replace("Ax","A") + out_ext)

    subprocess.run(f"cat {file_path} | cut -d ' ' -f{scol}-{ecol} | awk '{{$1=$1;print}}' > {out_path}", shell=True)

if get_coords:
    for tup in coord_vars:
        var = tup[0]
        print(f"Extract {var}.")

        col = tup[1]
        out_path = os.path.join(out_folder, var + out_ext)

        subprocess.run(f"cat {file_path} | cut -d ' ' -f{col} | awk '{{$1=$1;print}}' > {out_path}", shell=True)

print("Converted all data to edf format successfully.")