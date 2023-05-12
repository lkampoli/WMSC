import os

for folder in os.listdir():
    if "data_rnd" in folder:
        print(folder)
        for f in os.listdir(folder):
            if "T" == f[0]:
                print(f)
                os.rename(os.path.join(folder, f), os.path.join(folder, f.replace("T", "V")))
