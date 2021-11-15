from itertools import chain, islice
from torch.utils.data import Dataset, IterableDataset
from transformer.data.interface import DatasetInterface
from transformer.utils.common import read_txt, read_json

class DatasetFromObject(Dataset, DatasetInterface):
    def __init__(self, data, batch_size, shuffle=False, device=None, nprocs=1):
        self.data = data
        self.shuffle = shuffle
        self.data_size = len(self.data)
        DatasetInterface.__init__(self=self, batch_size=batch_size, device=device, nprocs=nprocs)

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        output = self.data[idx]
        return output

class DatasetFromDir(IterableDataset, DatasetInterface):
    def __init__(self, data_dir, batch_size, device="cpu", nprocs=1, encoding="utf-8", extension="json"):
        if not data_dir.endswith("/"): data_dir += "/"
        self.data_dir = data_dir
        self.file_path_list = self.get_file_path_list(data_dir=data_dir, extension=extension)
        self.encoding = encoding
        self.extension = extension
        self.data_size = self.count_lines()
        DatasetInterface.__init__(self=self, batch_size=batch_size, device=device, nprocs=nprocs)

    def count_lines(self):
        cnt = 0
        if self.extension == "txt":
            for file_path in self.file_path_list:
                rows = read_txt(path=file_path, encoding=self.encoding)
                cnt += len(rows)
        elif self.extension == "json":
            for file_path in self.file_path_list:
                rows = read_json(path=file_path, encoding=self.encoding)
                cnt += len(rows)
        return cnt

    def get_all_data(self):
        data = []
        if self.extension == "txt":
            for file_path in self.file_path_list:
                rows = read_txt(path=file_path, encoding=self.encoding)
                data += rows
        elif self.extension == "json":
            for file_path in self.file_path_list:
                rows = read_json(path=file_path, encoding=self.encoding)
                data += rows
        return data

    def get_stream(self, file_path_list, start, end):
        return islice(chain.from_iterable(map(self.parse_file, file_path_list)), start, end)

    def __len__(self):
        return self.data_size

    def __iter__(self):
        if not self.iter_range_update: self.set_iter_range()
        return self.get_stream(self.file_path_list, start=self.iter_start, end=self.iter_end)


class DatasetFromFile(IterableDataset, DatasetInterface):
    def __init__(self, file_path, batch_size, delimiter="\t", encoding="utf-8", extension="json", device=None, nprocs=1):
        self.file_path = file_path
        self.delimiter = delimiter
        self.encoding = encoding
        self.extension = extension
        self.data_size = self.count_lines()
        DatasetInterface.__init__(self=self, batch_size=batch_size, device=device, nprocs=nprocs)

    def count_lines(self):
        cnt = 0
        if self.extension == "txt":
            rows = read_txt(path=self.file_path, encoding=self.encoding)
            cnt += len(rows)
        elif self.extension == "json":
            rows = read_json(path=self.file_path, encoding=self.encoding)
            cnt += len(rows)
        return cnt

    def get_all_data(self):
        data = None
        if self.extension == "json":
            data = read_txt(path=self.file_path, encoding=self.encoding)
        elif self.extension == "json":
            data = read_json(path=self.file_path, encoding=self.encoding)
        return data

    def get_stream(self, file_path, start, end):
        return islice(self.parse_file(file_path), start, end)

    def __len__(self):
        return self.data_size

    def __iter__(self):
        if not self.iter_range_update: self.set_iter_range()
        return self.get_stream(self.file_path, start=self.iter_start, end=self.iter_end)

# class TransformerDataset(Dataset, DatasetInterface):
#     def __init__(self, preprocessor, data, **kwargs):
#         DatasetInterface.__init__(self, preprocessor=preprocessor, data=data)
#         required_parameters = ["src_timesteps", "tgt_timesteps"]
#         self.assert_contain_elements(required=required_parameters, target=kwargs, name="parameters")
#         self.__dict__.update(kwargs)
#         if "mask" not in kwargs: self.__dict__.update({"mask":False})
#         if "approach" not in kwargs: self.__dict__.update({"approach": "ignore"})
#         self._filter_data()
#
#     def __len__(self):
#         return self.data_size
#
#     def __getitem__(self, idx):
#         output = self.data[idx]
#         return output
#
#     def _filter_data(self):
#         output = []
#         for row in tqdm(self.data):
#             src_sentence, tgt_sentence = self._parse_row(row=row)
#             status, row = self.preprocessor.encode_row(src_sentence=src_sentence, src_timesteps=self.src_timesteps, tgt_sentence=tgt_sentence, tgt_timesteps=self.tgt_timesteps, mask=self.mask, approach=self.approach)
#             if status != 0: continue
#             row_dict = dict()
#             row_dict["src_sentences"] = src_sentence
#             row_dict["tgt_sentences"] = tgt_sentence
#             output.append(row_dict)
#
#         self.data = output
#         self.data_size = len(output)
#
#     def get_batch(self, from_idx, to_idx, device=None):
#         self.assert_not_out_of_index(index=from_idx, upperbound=self.data_size)
#         output = dict()
#
#         batch_data = self.data[from_idx:to_idx]
#         src_sentences = []
#         tgt_sentences = []
#         for row in batch_data:
#             src_sentence = row["src_sentences"]
#             tgt_sentence = row["tgt_sentences"]
#             src_sentences.append(src_sentence)
#             tgt_sentences.append(tgt_sentence)
#
#         src_inputs, tgt_inputs, tgt_labels = self.preprocessor.encode(src_sentences=src_sentences, src_timesteps=self.src_timesteps, tgt_sentences=tgt_sentences, tgt_timesteps=self.tgt_timesteps, mask=self.mask, approach=self.approach)
#         src_inputs = torch.from_numpy(np.array(src_inputs))
#         tgt_inputs = torch.from_numpy(np.array(tgt_inputs))
#         tgt_labels = torch.from_numpy(np.array(tgt_labels, dtype=np.int64))
#
#         if device is not None:
#             src_inputs = src_inputs.to(device, non_blocking=True)
#             tgt_inputs = tgt_inputs.to(device, non_blocking=True)
#             tgt_labels = tgt_labels.to(device, non_blocking=True)
#
#         output["src_inputs"] = src_inputs
#         output["tgt_inputs"] = tgt_inputs
#         output["tgt_labels"] = tgt_labels
#         return output
#
#     def _parse_row(self, row):
#         src_sentence, tgt_sentence = row.split("\t")
#         return src_sentence, tgt_sentence