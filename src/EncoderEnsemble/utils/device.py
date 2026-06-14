import torch


def choose_device(allow_cpu):
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU:", torch.cuda.get_device_name(0))
        return device
    if allow_cpu:
        print("Using CPU because --allow-cpu was set.")
        return torch.device("cpu")
    raise RuntimeError("CUDA GPU is not available. Use a CUDA kernel or pass --allow-cpu for debugging.")
