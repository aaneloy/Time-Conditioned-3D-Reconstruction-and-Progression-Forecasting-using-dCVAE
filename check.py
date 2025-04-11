import os

root = "data/datasets/brats2021"

print(f"[DEBUG] Absolute path: {os.path.abspath(root)}")
print(f"[DEBUG] Exists? {os.path.exists(root)}")

if os.path.exists(root):
    contents = os.listdir(root)
    print(f"[DEBUG] Items inside '{root}':")
    for item in contents:
        print(" -", item)
else:
    print("[ERROR] Directory does NOT exist!")
