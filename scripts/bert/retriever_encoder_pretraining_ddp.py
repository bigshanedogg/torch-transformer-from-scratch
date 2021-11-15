import os
import sys
import json
import torch
import argparse
sys.path.append(os.path.abspath('./'))
from transformer.models.bert import Bert
from transformer.preprocessors.blender_bot_preprocessor import RetrieverEncoderPreprocessor
from transformer.trainers.blender_bot_trainer import RetrieverEncoderBertTrainer
from transformer.utils.common import read_all_files_from_dir, set_seed
from setproctitle import setproctitle
setproctitle("Bert_DDP")

config = None
ddp_config = None
train_config = None
data_config = None
model_config = None
optimizer_config = None
criterion_config = None
data_loader_config = None

def parse():
    parser = argparse.ArgumentParser(description="Pre-train Bert on DDP")
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
    trainer = RetrieverEncoderBertTrainer(temp_dir=train_config["temp_save_path"])
    trainer.set_ngpus_per_node(ngpus_per_node=ngpus_per_node)
    if train_config["lr_update"]:
        trainer.set_lr_update(initial_learning_rate=optimizer_config["initial_learning_rate"], num_warmup_steps=train_config["num_warmup_steps"])

    # train spm_model if not exists
    spm_model_path = data_config["spm_model_path"].format(root_dir=data_config["root_dir"], language=data_config["language"], vocab_size=model_config["vocab_size"])

    # define bert_preprocessor
    print('From Global ==> Define preprocessor..')
    preprocessor = RetrieverEncoderPreprocessor(language=data_config["language"], spm_model_path=spm_model_path, embedding_dict=model_config["embedding_dict"])

    # get arguments to pass 'main_worker' method
    print('From Global ==> Setting model_params..')
    model_params = trainer.get_init_params(timesteps=model_config["timesteps"], vocab_size=model_config["vocab_size"], embedding_dict=model_config["embedding_dict"],
                                        d_model=model_config["d_model"], d_ff=model_config["d_ff"], num_heads=model_config["num_heads"], num_layers=model_config["num_layers"],
                                        shared_embedding=model_config["shared_embedding"], pad_token_id=preprocessor.spm_tokenizer.special_token_dict["pad"]["id"],
                                        dropout=model_config["dropout"], pwff_activation=model_config["pwff_activation"],
                                        linear_activation=model_config["linear_activation"], bias=model_config["bias"],
                                        layer_norm_epsilon=model_config["layer_norm_epsilon"], initialization=model_config["initialization"])

    print('From Global ==> Setting arguments for main_worker..')
    ddp_config["nprocs"] = nprocs
    ddp_params = RetrieverEncoderBertTrainer.get_ddp_params(**ddp_config)
    train_params = RetrieverEncoderBertTrainer.get_train_params(**train_config)
    data_config["train_data_dir"] = data_config["train_data_dir"].format(root_dir=data_config["root_dir"])
    # data_config["val_data_dir"] = data_config["val_data_dir"].format(root_dir=data_config["root_dir"])
    data_config["val_data_dir"] = None
    data_params = RetrieverEncoderBertTrainer.get_data_params(**data_config)
    data_loader_config["timesteps"] = model_config["timesteps"]
    data_loader_config["embedding_dict"] = model_config["embedding_dict"]

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
                                      model_params, optimizer_config, criterion_config, data_loader_config,))
    ############################################################

if __name__ == '__main__':
    main()