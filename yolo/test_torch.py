import torch

print(torch.cuda.is_available())  # Should return True if CUDA is correctly installed
print(torch.version.cuda)         # To check the CUDA version
print(torch.backends.cudnn.enabled)  # To confirm cuDNN is enabled
