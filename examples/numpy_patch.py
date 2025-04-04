import numpy as np

# Add bool8 as an alias for bool if it doesn't exist
if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_

print("NumPy patched successfully: bool8 is now available as an alias for bool_")
