import os
import shutil

def organize(src_folder, output_name, limit=300):
    output_path= os.path.join(src_folder, output_name)
    os.makedirs(output_path)
    count = 1
    for root, dir, files in os.walk(src_folder):
        for fname in files:
            if fname.lower().endswith(('.jpeg', '.jpg', '.png')):
                if count > limit:
                    return
                f1 = os.path.join(root, fname)
                new_fname = f"{count}.jpeg"
                f2 = os.path.join(output_path, new_fname)
                shutil.copy2(f1, f2)
                count += 1

#you can change those values 
src = '1_BillNER' 
target = 'Selected' 
organize(src, target, limit=300)