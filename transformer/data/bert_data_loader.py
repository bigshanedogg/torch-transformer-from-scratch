from transformer.data.interface import DataLoaderInterface

class BertDataLoader(DataLoaderInterface):
    def __init__(self, dataset, preprocessor, batch_size, device, nprocs, num_workers, pin_memory,
                 timesteps, embedding_dict, sep_tokens, approach, make_negative_sample):
        DataLoaderInterface.__init__(self=self, dataset=dataset, preprocessor=preprocessor, batch_size=batch_size,
                                     device=device, nprocs=nprocs, num_workers=num_workers, pin_memory=pin_memory)
        self.timesteps = timesteps
        self.embedding_dict = embedding_dict
        self.sep_tokens = sep_tokens
        self.approach = approach
        self.make_negative_sample = make_negative_sample
        # assert
        self.preprocessor.spm_tokenizer.assert_isloaded_spm_model()
        self.preprocessor.assert_isin_approaches(approach=self.approach)