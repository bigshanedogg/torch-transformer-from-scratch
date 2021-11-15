import torch
from transformer.assertions.object_assertion import ServiceAssertion
from transformer.utils.common import get_device_index, get_available_devices

class ServiceInterface(ServiceAssertion):
    device = "cpu"
    trainer = None
    model = None
    model_history = None
    preprocessor = None
    # constants
    temp_dir = None
    available_devices = []
    available_device_indice = []

    def __init__(self, temp_dir="./"):
        # constant
        self.temp_dir = temp_dir
        self.available_devices = get_available_devices()
        self.available_device_indice = [get_device_index(device) for device in self.available_devices]
        # instance setting
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.set_trainer(temp_dir=temp_dir)

    def set_trainer(self, temp_dir):
        self.assert_implemented(method_name="set_trainer")

    def set_device(self, device:str):
        device_index = get_device_index(device)
        if device_index in self.available_device_indice:
            self.device = "cuda:{device_index}".format(device_index=device_index)
        else:
            self.device = "cpu"
        return self.device

    def load_model(self, model_dir, language="kor"):
        self.assert_implemented(method_name="load_model")