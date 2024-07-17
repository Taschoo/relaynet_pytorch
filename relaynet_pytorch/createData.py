import numpy as np
import h5py

# Create dummy data for 'Data.h5'
data = np.random.rand(100, 640, 480).astype(np.float32)
with h5py.File('datasets/Data.h5', 'w') as f:
    f.create_dataset('data', data=data)

# Create dummy labels for 'label.h5'
labels = np.random.randint(0, 9, size=(100, 2, 640, 480)).astype(np.int64)
with h5py.File('datasets/label.h5', 'w') as f:
    f.create_dataset('label', data=labels)

# Create dummy set definitions for 'set.h5'
set_ids = np.random.choice([1, 3], size=100).astype(np.int64)
with h5py.File('datasets/set.h5', 'w') as f:
    f.create_dataset('set', data=set_ids)
