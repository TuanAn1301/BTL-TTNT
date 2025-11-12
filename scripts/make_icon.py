from PIL import Image
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src = os.path.join(BASE_DIR, 'gui', 'logo', 'logo.png')
dst = os.path.join(BASE_DIR, 'gui', 'logo', 'logo.ico')

if not os.path.exists(src):
    print(f"ERROR: Source PNG not found: {src}")
    sys.exit(1)

os.makedirs(os.path.dirname(dst), exist_ok=True)

im = Image.open(src).convert('RGBA')
# Create multi-size ICO
sizes = [(256,256),(128,128),(64,64),(48,48),(32,32),(16,16)]
im.save(dst, sizes=sizes)
print(f"OK: created {dst}")
