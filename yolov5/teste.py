from copy import deepcopy

class add2:
  def __init__(self, bias):
    self.bias = bias 
  def __call__(self, inp):
    return inp + 2 + self.bias

class mul3:
  def __init__(self, bias):
    self.bias = bias 
  def __call__(self, inp):
    return inp * 3 + self.bias

def sequential(*args):
  x = 0
  for f in args:
    print(id(f))
    x = f(x)
  return x

x1 = sequential(add2(1), add2(1), mul3(1))
print()
argss = [add2(1) for _ in range(2)]
x2 = sequential(*argss, mul3(1))

print(x1, x2)
