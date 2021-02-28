import torch
a = torch.ones(5)

a.requires_grad = True

b = 2*a

b.retain_grad()

b.register_hook(lambda x: print(x))

b.mean().backward()


print(a.grad, b.grad)
