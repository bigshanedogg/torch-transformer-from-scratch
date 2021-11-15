import os
import sys
import json
import torch
import argparse
sys.path.append(os.path.abspath('./'))
from transformer.data.dataset import DatasetFromDir
from transformer.models.transformer import Transformer
from transformer.preprocessors.transformer_preprocessor import TransformerPreprocessor
from transformer.preprocessors.blender_bot_preprocessor import GeneratorPretrainingPreprocessor
from transformer.trainers.transformer_trainer import TransformerTrainer
from transformer.trainers.blender_bot_trainer import GeneratorPretrainingTransformerTrainer
from transformer.utils.common import read_all_files_from_dir, set_seed
from setproctitle import setproctitle
setproctitle("Transformer_DDP")

config = None
ddp_config = None
train_config = None
data_config = None
model_config = None
optimizer_config = None
criterion_config = None
data_loader_config = None

def parse():
    parser = argparse.ArgumentParser(description="Pre-train Transformer on DDP")
    parser.add_argument("--config_path", metavar="Config json file path", help="specify config json path; e.g. './scripts/transformer/config/retriever_finetuning_korea.json") # "./scripts/transformer/config/retriever_finetuning_korea.json"
    args = parser.parse_args()
    return args

def main():
    # parse arguments
    args = parse()
    with open(args.config_path, "r", encoding="UTF-8") as fp:
        config = json.load(fp)
        ddp_config = config["ddp"]
        train_config = config["train"]
        data_config = config["data"]
        model_config = config["model"]
        optimizer_config = config["optimizer"]
        criterion_config = config["criterion"]
        data_loader_config = config["data_loader"]

    # set_seed
    set_seed(train_config["seed"])

    print('From Global ==> Define trainer..')
    gpu_devices = ddp_config["gpu_devices"].split(",")
    nprocs = ngpus_per_node = min(len(gpu_devices), torch.cuda.device_count())
    trainer = GeneratorPretrainingTransformerTrainer(temp_dir=train_config["temp_save_path"])
    trainer.set_ngpus_per_node(ngpus_per_node=ngpus_per_node, gpu_devices=gpu_devices)

    if train_config["lr_update"]:
        trainer.set_lr_update(initial_learning_rate=optimizer_config["initial_learning_rate"], num_warmup_steps=train_config["num_warmup_steps"])

    # load spm_model
    src_spm_model_path = data_config["src_spm_model_path"].format(root_dir=data_config["root_dir"], language=data_config["src_language"], vocab_size=model_config["src_vocab_size"])
    tgt_spm_model_path = data_config["src_spm_model_path"].format(root_dir=data_config["root_dir"], language=data_config["tgt_language"], vocab_size=model_config["tgt_vocab_size"])

    # define preprocessor
    print('From Global ==> Define preprocessor..')
    preprocessor = GeneratorPretrainingPreprocessor(src_language=data_config["src_language"], tgt_language=data_config["tgt_language"],
                                                    src_spm_model_path=src_spm_model_path, tgt_spm_model_path=tgt_spm_model_path, embedding_dict=model_config["embedding_dict"])

    # get arguments to pass 'main_worker' method
    print('From Global ==> Setting model_params..')
    model_params = trainer.get_init_params(src_timesteps=model_config["src_timesteps"], tgt_timesteps=model_config["tgt_timesteps"],
                                           src_vocab_size=model_config["src_vocab_size"], tgt_vocab_size=model_config["tgt_vocab_size"], embedding_dict=model_config["embedding_dict"],
                                           src_pad_token_id=preprocessor.src_spm_tokenizer.special_token_dict["pad"]["id"],
                                           tgt_pad_token_id=preprocessor.tgt_spm_tokenizer.special_token_dict["pad"]["id"],
                                           d_model=model_config["d_model"], d_ff=model_config["d_ff"], num_heads=model_config["num_heads"],
                                           num_encoder_layers=model_config["num_encoder_layers"], num_decoder_layers=model_config["num_decoder_layers"],
                                           shared_embedding=model_config["shared_embedding"], dropout=model_config["dropout"],
                                           pwff_activation=model_config["pwff_activation"], linear_activation=model_config["linear_activation"],
                                           bias=model_config["bias"], layer_norm_epsilon=model_config["layer_norm_epsilon"], initialization=model_config["initialization"])

    print('From Global ==> Setting arguments for main_worker..')
    ddp_config["nprocs"] = nprocs
    ddp_params = GeneratorPretrainingTransformerTrainer.get_ddp_params(**ddp_config)
    train_params = GeneratorPretrainingTransformerTrainer.get_train_params(**train_config)
    data_config["train_data_dir"] = data_config["train_data_dir"].format(root_dir=data_config["root_dir"])
    data_config["val_data_dir"] = data_config["val_data_dir"].format(root_dir=data_config["root_dir"])
    # data_config["val_data_dir"] = None
    data_params = GeneratorPretrainingTransformerTrainer.get_data_params(**data_config)
    data_loader_config["src_timesteps"] = model_config["src_timesteps"]
    data_loader_config["tgt_timesteps"] = model_config["tgt_timesteps"]
    data_loader_config["embedding_dict"] = model_config["embedding_dict"]

    # set pre_token_distribution for ulk_loss
    if "ul" in criterion_config and criterion_config["ul"] > 0.0:
        print('From Global ==> Set prev_token_distribution for ulk_loss..')
        train_dataset = DatasetFromDir(data_dir=data_config["train_data_dir"], batch_size=train_config["batch_size"], device="cpu", nprocs=nprocs, encoding=data_config["encoding"], extension=data_config["extension"])
        utterances = [utterance for row in train_dataset.get_all_data() for utterance in row["utterances"]]
        target_prev_token_distribution, special_token_ids = preprocessor.extract_prev_token_distribution(sentences=utterances, ngram=criterion_config["ngram"])
        trainer.set_prev_token_distribution(prev_token_distribution=target_prev_token_distribution, special_token_ids=special_token_ids)

    ############################################################
    # on each node we have: ngpus_per_node processes and ngpus_per_node gpus
    # that is, 1 process for each gpu on each node.
    # node <- process <- gpu
    # world_size, nprocs: the number of process to run
    # ngpus_per_node: the number of gpu devices per node
    # rank: index of node
    # Use torch.multiprocessing.spawn to launch distributed processes: the main_worker process function
    print('From Global ==> Start spawn..')
    torch.multiprocessing.spawn(fn=trainer.main_worker, nprocs=nprocs,
                                args=(config, preprocessor, ddp_params, train_params, data_params,
                                      model_params, optimizer_config, criterion_config, data_loader_config, ))
    ############################################################

if __name__ == '__main__':
    main()