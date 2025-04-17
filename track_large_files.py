import os
import subprocess

limit_bytes = 5 * 1024 * 1024  # 5MB

for root, dirs, files in os.walk("."):
    for file in files:
        path = os.path.join(root, file)
        if os.path.getsize(path) > limit_bytes:
            print(f"Tracking {path}")
            subprocess.run(["git", "lfs", "track", path.replace("\\", "/")])
