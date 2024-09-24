from pathlib import Path
from PIL import Image
import numpy as np

N_CLS = 10
N_IMGS = 10


def mock_dataset(func):
    def wrapper_mock_dataset(*args, **kwargs):
        # create a mock dataset
        root = Path("./tmp/mock_dataset")
        root.mkdir(parents=True, exist_ok=True)
        for i in range(N_CLS):
            cls_dir = root / str(i)
            cls_dir.mkdir(parents=True, exist_ok=True)
            for j in range(N_IMGS):
                img = Image.fromarray(np.random.randint(0, 255, size=(32, 32, 3)).astype(np.uint8))
                img.save(cls_dir / f"{j}.png")

        # call the decorated function
        kwargs["root"] = root
        out = func(*args, **kwargs)

        # cleanup
        for i in range(10):
            cls_dir = root / str(i)
            for img_path in cls_dir.iterdir():
                img_path.unlink()
            cls_dir.rmdir()

        return out
        
    return wrapper_mock_dataset