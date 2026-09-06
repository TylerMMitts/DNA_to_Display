# Runs the LiteVAE demo on one image from the project root.
#
# A thin wrapper around litevae/demo.py so the demo can be started without
# knowing the package layout.

import os
import sys
from pathlib import Path

# Puts code/ on the import path so this file can be run directly by path.
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))

from paths import PROJECT_ROOT, RESULTS_DIR

from litevae.demo import run_litevae_demo

if __name__ == "__main__":
    # Anchored to the project root rather than the working directory, so the
    # demo picks up the same image wherever it is launched from.
    image_path = PROJECT_ROOT / 'image.JPG'
    save_dir = RESULTS_DIR / 'litevae_output'

    if not image_path.exists():
        print("Creating dummy image for demonstration...")
        import numpy as np
        from PIL import Image
        dummy = np.random.rand(256, 256, 3) * 255
        Image.fromarray(dummy.astype(np.uint8)).save(image_path)

    z, recon = run_litevae_demo(image_path, save_dir)

    print(f"Check {save_dir} for all saved intermediate steps.")
