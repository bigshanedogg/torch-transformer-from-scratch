import os
import re
import json
import random
import shutil
import numpy as np
import torch
from datetime import datetime

def show_five_nums(data, verbose=True):
    quartiles = np.percentile(data, [25, 50, 75])
    min_v = np.min(data)
    max_v = np.max(data)
    avg = np.mean(data)
    if verbose:
        print("Min: {min_v:.3f}\tMax: {max_v:.3f}\tAvg: {avg:.3f}\tQ1: {q1:.3f}\tQ2: {q2:.3f}\tQ3: {q3:.3f}".format(min_v=min_v, max_v=max_v, avg=avg, q1=quartiles[0], q2=quartiles[1], q3=quartiles[2]))
    return min_v, max_v, quartiles[0], quartiles[1], quartiles[2], avg

def is_primitive(obj):
    return not hasattr(obj, '__dict__')

def get_now_str(str_format="%Y%m%d_%H%M%S"):
    now_str = datetime.now().strftime(str_format)
    return now_str

def get_nth_index(obj, value, n):
    indices = [idx for idx, element in enumerate(obj) if element == value]
    if len(indices) == 0 or len(indices) < n:
        return -1
    else:
        return indices[n]

def get_last_index(obj, value):
    indices = [idx for idx, element in enumerate(obj) if element == value]
    if len(indices) == 0:
        return -1
    else:
        return indices[-1]

def get_randint_except(low, high, except_value):
    assert except_value in range(low, high), "'except_value' must be between low and high."
    assert high > low, "high must be greater than low."
    assert not (except_value == low and low + 1 == high), \
        "except_value cannot be the only value in range: except_value:{ev}, low:{l}, high:{h".format(ev=except_value, l=low, h=high)
    value = np.random.randint(low=low, high=high)
    while value == except_value:
        value = np.random.randint(low=low, high=high)
    return value

def shuffle_related_lists(lists):
    is_equal_length = True
    length = len(lists[0])
    for i in range(1, len(lists)):
        if length != len(lists[i]):
            is_equal_length = False
            break
    assert is_equal_length, "All lists must have equal lengthes"

    shuffled = [[] for _ in range(0, len(lists))]
    indices = list(range(0, length))
    random.shuffle(indices)
    for idx in indices:
        for i, _list in enumerate(lists):
            shuffled[i].append(_list[idx])
    return shuffled

def shuffle_dictionary_lists(dictionaries):
    is_equal_length = True
    length = -1
    for dictionary in dictionaries:
        for k,v in dictionary.items():
            if length < 0: length = len(v)
            if length != len(v):
                is_equal_length = False
                break
        if not is_equal_length: break
    assert is_equal_length, "All lists must have equal lengthes"

    shuffled = [{k:[] for k,v in dictionary.items()} for dictionary in dictionaries]
    indices = list(range(0, length))
    random.shuffle(indices)
    for dictionary_idx, dictionary in enumerate(dictionaries):
        for k,v in dictionary.items():
            shuffled[dictionary_idx][k] = [v[idx] for idx in indices]
    return shuffled

def lambda_lr_transformer(current_step, num_warmup_steps, d_model):
    arg1 = (current_step + 1) ** -0.5
    arg2 = (current_step + 1) * (num_warmup_steps ** -1.5)
    lr = (d_model ** -0.5) * min(arg1, arg2)
    return lr

def read_all_files_from_dir(data_dir, extension, encoding="utf-8"):
    data = []
    file_path_list = os.listdir(data_dir)
    for file_path in file_path_list:
        if not file_path.endswith(extension): continue
        with open(data_dir+file_path, "r", encoding=encoding) as fp:
            if extension.endswith("txt"):
                for row in fp:
                    data.append(row)
            elif extension.endswith("json"):
                rows = json.load(fp)
                data += rows
    return data

def is_empty_row_in_dict(data):
    mask = [len(v)<=0 for k, v in data.items()]
    flag = any(mask)
    return flag

def is_valid_file(path):
    if not os.path.exists(path): return False
    if not os.path.isfile(path): return False
    return True

def is_valid_path(path):
    if not os.path.exists(path): return False
    if not os.path.isdir(path): return False
    return True

def init_path(path, reset=False):
    if os.path.exists(path) and reset: shutil.rmtree(path)
    if not os.path.exists(path): os.makedirs(path)
    if not path.endswith("/"): path += "/"
    return path

def is_cpu_device(device):
    if isinstance(device, str) and device == "cpu":
        return True
    elif isinstance(device, torch.device) and device.type == "cpu":
        return True
    else:
        return False

def get_device_index(device, default=0):
    device_index = default
    if is_cpu_device(device):
        device_index = default
    elif isinstance(device, torch.device):
        device_index = device.index
    elif isinstance(device, str):
        if re.search("[a-zA-Z]", device) is not None:
            device = re.sub("[^0-9]", "", device)
        device_index = int(device)
    elif isinstance(device, int):
        device_index = device
    return device_index

def get_available_devices():
    devices = []
    if torch.cuda.is_available():
        num_cuda_devices = torch.cuda.device_count()
        for device_index in range(0, num_cuda_devices):
            try:
                _capability = torch.cuda.get_device_capability(device_index)
                device = "cuda:{idx}".format(idx=device_index)
                devices.append(device)
            except AssertionError as ex:
                continue
    return devices

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def read_txt(path, encoding):
    data = []
    with open(path, "r", encoding=encoding) as fp:
        for row in fp:
            data.append(row)
    return data

def read_json(path, encoding):
    data = []
    with open(path, "r", encoding=encoding) as fp:
        data = json.load(fp)
    return data

def split_segment_by_turn(utterances, turn_ids):
    sequence = []
    segment = []
    first_turn_id = turn_ids[0]
    replied = False
    for idx, (turn_id, utterance) in enumerate(zip(turn_ids, utterances)):
        if replied and turn_id == first_turn_id:
            sequence.append(segment)
            segment = []
            replied = False

        segment.append(utterance)
        if turn_id != first_turn_id: replied = True
    sequence.append(segment)
    return sequence

def get_top_n_values(values, top_n=5, descending=True):
    output = [(idx, _score) for idx, _score in enumerate(values)]
    if descending:
        output = sorted(output, key=lambda x: -x[1])
    else:
        output = sorted(output, key=lambda x: x[1])
    if top_n > 0:
        output = output[:top_n]
    return output