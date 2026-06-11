import numpy as np
from PIL import Image
import os

print("Lade eng_xacby.bin in den Speicher...")
# Float32 Daten im Row-Major Format lesen
data = np.fromfile('eng_xacby.bin', dtype=np.float32)

# Tensor hat das Format: [x=512, a=16, c=16, b=16, y=512]
data = data.reshape((512, 16, 16, 16, 512))

output_dir = 'output_pngs'
os.makedirs(output_dir, exist_ok=True)

print("Schreibe 9x9 Light Field Views als PNGs...")

# Das tatsächliche Camera-Array nutzt 9x9 Bilder. (Range 0-8)
for a in range(9):
    for b in range(9):
        # 1. Wir extrahieren die Spacial-Ebenen x und y, sowie die ersten 3 Kanäle (RGB) für 'c'
        # Shape hieraus resultierend: (512(x), 3(c), 512(y))
        view = data[:, a, 0:3, b, :]
        
        # 2. Die Standard PIL Repräsentation ist [Height(y), Width(x), Channels(c)]
        # Daher transponieren wir das Layout (x, c, y) auf (y, x, c)
        img = np.transpose(view, (2, 0, 1))
        
        # 3. Float RGB Werte auf Bereich von [0, 1] clippen und mit 255 skaliert abspeichern
        img = np.clip(img, 0.0, 1.0)
        img_u8 = (img * 255.0).astype(np.uint8)
        
        pil_img = Image.fromarray(img_u8, mode='RGB')
        pil_img.save(f'{output_dir}/view_a{a}_b{b}.png')

print(f"Abgeschlossen! Es wurden 81 Bilder im Ordner '{output_dir}/' gespeichert.")