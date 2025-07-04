
def to_cuda(module, device=None):
    """Assume module in cpu"""
    if device is None:
        return module.cuda()
    elif device == 'cpu':
        return module
    else:
        return module.to(device)
      
def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module
