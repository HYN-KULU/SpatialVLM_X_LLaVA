import torch
import torch.nn as nn
m = nn.Softmax(dim=0)
input = torch.randn(2, 3)
output = m(input)
out=torch.load("./save_llava.pth")
print(m(out["logits"][0][-1])[5852])