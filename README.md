# pytorch-code-snippets
Some useful pytorch code snippets

## save model and load model
[SAVING AND LOADING MODELS](https://pytorch.org/tutorials/beginner/saving_loading_models.html)

save model: 
```python
import torch
import torchvision.models as models

# Use an existing model from Torchvision, note it 
# will download this if not already on your computer (might take time)
model = models.shufflenet_v2_x0_5(pretrained=True)
torch.save(model.state_dict(), "shufflenet.pth")
```

load model:
```py
import torch
import torchvision.models as models

model = models.shufflenet_v2_x0_5()
model.load_state_dict(torch.load("shufflenet.pth"))
model.eval()
print(model)
```

## export model to onnx
[convert-pytorch-onnx](https://michhar.github.io/convert-pytorch-onnx/)
```python
import torch
import torchvision.models as models

model = models.shufflenet_v2_x0_5(pretrained=True)

# Create some sample input in the shape this model expects
# the expected input size can be found at https://pytorch.org/hub/pytorch_vision_shufflenet_v2/
dummy_input = torch.randn(1, 3, 224, 224)

# Use the exporter from torch to convert to onnx 
# model (that has the weights and net arch)
torch.onnx.export(model, dummy_input, "shufflenet.onnx", verbose=True)
```

## convert into one-hot encoding
[Convert int into one-hot format](https://discuss.pytorch.org/t/convert-int-into-one-hot-format/507/4)
```py
import torch

batch_size = 4
num_classes = 6
# Dummy input that HAS to be 2D for the scatter (you can use view(-1,1) if needed)
y = torch.LongTensor(batch_size,1).random_() % num_classes
"""
tensor([[1],
        [4],
        [5],
        [3]])
"""
# One hot encoding buffer that you create out of the loop and just keep reusing
y_onehot = torch.FloatTensor(batch_size, num_classes)
"""
tensor([[0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.0000, 0.0000, 0.0000, 1.8750, 0.0000, 0.0000]])
"""

# In your for loop
y_onehot.zero_()
"""
tensor([[0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0.]])
"""
# scatter_(dim, index, src)
y_onehot.scatter_(1, y, 1)
"""
tensor([[0., 1., 0., 0., 0., 0.],
        [0., 0., 0., 0., 1., 0.],
        [0., 0., 0., 0., 0., 1.],
        [0., 0., 0., 1., 0., 0.]])
"""
```
