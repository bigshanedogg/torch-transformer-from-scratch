import os
import sys
import json
import torch
import argparse
sys.path.append(os.path.abspath('./'))
from transformer.models.bert import Bert
from transformer.models.poly_encoder import PolyEncoder
from transformer.preprocessors.blender_bot_preprocessor import RetrieverFinetuningPreprocessor
from transformer.trainers.blender_bot_trainer import RetrieverFinetuningPolyEncoderTrainer
from transformer.utils.common import read_all_files_from_dir, set_seed
from transformer.trainers.utils import ModelFilenameConstants, load_state_dict, is_model_saved
from setproctitle import setproctitle
setproctitle("PolyEncoder_DDP")

config = None
ddp_config = None
train_config = None
data_config = None
model_config = None
optimizer_config = None
criterion_config = None
data_loader_config = None


def parse():
    parser = argparse.ArgumentParser(description="Fine-tuning PolyEncoder on DDP")
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

    initial_train = True
    if is_model_saved(path=train_config["resume_path"], save_hyperparms=False):
        initial_train = False

    model_config["context_encoder"]["model_path"] = model_config["context_encoder"]["model_path"].format(root_dir=data_config["root_dir"])
    model_config["candidate_encoder"]["model_path"] = model_config["candidate_encoder"]["model_path"].format(root_dir=data_config["root_dir"])
    context_encoder_model_path = model_config["context_encoder"]["model_path"]
    with open(context_encoder_model_path + ModelFilenameConstants.TRAIN_CONFIG_FILENAME, "r", encoding="UTF-8") as fp:
        context_encoder_config = json.load(fp)

    # set_seed
    set_seed(train_config["seed"])

    print('From Global ==> Define trainer..')
    gpu_devices = ddp_config["gpu_devices"].split(",")
    nprocs = ngpus_per_node = min(len(gpu_devices), torch.cuda.device_count())
    trainer = RetrieverFinetuningPolyEncoderTrainer(temp_dir=train_config["temp_save_path"])
    trainer.set_ngpus_per_node(ngpus_per_node=ngpus_per_node)
    if train_config["lr_update"]:
        trainer.set_lr_update(initial_learning_rate=optimizer_config["initial_learning_rate"], num_warmup_steps=train_config["num_warmup_steps"])

    # define bert_preprocessor
    print('From Global ==> Define preprocessor..')
    preprocessor = None
    if initial_train:
        spm_model_path = context_encoder_model_path + ModelFilenameConstants.SPM_MODEL_DIR
        preprocessor = RetrieverFinetuningPreprocessor(language=data_config["language"], spm_model_path=spm_model_path, embedding_dict=context_encoder_config["model"]["embedding_dict"])
    else:
        spm_model_path = train_config["resume_path"] + ModelFilenameConstants.SPM_MODEL_DIR
        preprocessor = RetrieverFinetuningPreprocessor(language=data_config["language"], spm_model_path=spm_model_path, embedding_dict=context_encoder_config["model"]["embedding_dict"])

    # get arguments to pass 'main_worker' method
    print('From Global ==> Setting model_params..')
    model_params = trainer.get_init_params(**model_config)

    print('From Global ==> Setting arguments for main_worker..')
    ddp_config["nprocs"] = nprocs
    ddp_params = RetrieverFinetuningPolyEncoderTrainer.get_ddp_params(**ddp_config)
    train_params = RetrieverFinetuningPolyEncoderTrainer.get_train_params(**train_config)
    data_config["train_data_dir"] = data_config["train_data_dir"].format(root_dir=data_config["root_dir"])
    data_config["val_data_dir"] = data_config["val_data_dir"].format(root_dir=data_config["root_dir"])
    data_params = RetrieverFinetuningPolyEncoderTrainer.get_data_params(**data_config)
    data_loader_config["timesteps"] = context_encoder_config["model"]["timesteps"]
    data_loader_config["embedding_dict"] = context_encoder_config["model"]["embedding_dict"]

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