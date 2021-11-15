from transformer.data.interface import DataLoaderInterface

class TransformerDataLoader(DataLoaderInterface):
    def __init__(self, dataset, preprocessor, batch_size, device, nprocs, num_workers, pin_memory,
                 src_timesteps, tgt_timesteps, embedding_dict, src_sep_tokens, approach):
        DataLoaderInterface.__init__(self=self, dataset=dataset, preprocessor=preprocessor, batch_size=batch_size,
                                     device=device, nprocs=nprocs, num_workers=num_workers, pin_memory=pin_memory)
        self.src_timesteps = src_timesteps
        self.tgt_timesteps = tgt_timesteps
        self.embedding_dict = embedding_dict
        self.src_sep_tokens = src_sep_tokens
        self.approach = approach
        # assert
        self.preprocessor.src_spm_tokenizer.assert_isloaded_spm_model()
        self.preprocessor.tgt_spm_tokenizer.assert_isloaded_spm_model()
        self.preprocessor.assert_isin_approaches(approach=self.approach)