# pytorch-code-snippets
Some useful pytorch code snippets

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
