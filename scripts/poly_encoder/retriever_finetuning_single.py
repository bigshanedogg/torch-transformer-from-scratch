import os
import sys
import json
import argparse
import torch
sys.path.append(os.path.abspath('./'))
from transformer.data.dataset import DatasetFromObject, DatasetFromDir, DatasetFromFile
from transformer.preprocessors.sentence_bert_preprocessor import SentenceBertPreprocessor
from transformer.models.bert import Bert
from transformer.models.poly_encoder import PolyEncoder
from transformer.trainers.interface import TrainerInterface
from transformer.trainers.blender_bot_trainer import RetrieverFinetuningPolyEncoderTrainer
from transformer.trainers.utils import ModelFilenameConstants, save_model, save_optimizer, save_history, load_state_dict, is_model_saved, is_optimizer_saved
from transformer.utils.common import set_seed, get_device_index
from setproctitle import setproctitle
setproctitle("PolyEncoder_Single")

config = None
with open("./scripts/transformer/config/retriever_finetuning_korea.json", "r", encoding="UTF-8") as fp:
    config = json.load(fp)
train_config = config["train"]
model_config = config["model"]
data_config = config["data"]
data_loader_config = config["data_loader"]

def parse():
    parser = argparse.ArgumentParser(description="Fine-Tuning PolyEncoder on Single-GPU")
    parser.add_argument("--cuda", metavar="CUDA", default="0", help="specify cuda device")
    parser.add_argument("--train_data_dir", metavar="Train Data Directory", default=None, help="specify train_data_dir")
    parser.add_argument("--val_data_dir", metavar="Validation Data Directory", default=None, help="specify val_data_dir")
    args = parser.parse_args()
    return args

def main():
    # parse arguments
    args = parse()
    device = torch.device("cuda:{cuda}".format(cuda=args.cuda) if torch.cuda.is_available() else "cpu")
    device_index = get_device_index(device=device)
    train_data_dir = args.train_data_dir
    if train_data_dir is None: data_config["train_data_dir"]
    nprocs = 1

    # set_seed
    set_seed(train_config["seed"])

    # load spm_model
    encoder_model_path = train_config["encoder_model_path"]
    encoder_config_path = encoder_model_path + ModelFilenameConstants.TRAIN_CONFIG_FILENAME
    encoder_model_state_dict_path = encoder_config_path + ModelFilenameConstants.MODEL_STATE_DICT_FILENAME
    encoder_spm_model_path = encoder_model_path + ModelFilenameConstants.SPM_MODEL_DIR
    with open(encoder_config_path, "r", encoding="utf-8") as fp:
        encoder_config = json.load(fp)

    # define preprocessor
    print('From GPU Device {} ==> Define preprocessor..'.format(device_index))
    preprocessor = SentenceBertPreprocessor(src_language=data_config["language"], spm_model_path=encoder_spm_model_path, embedding_dict=model_config["embedding_dict"])

    # load context_encoder & candidate_encoder
    print('From GPU Device {} ==> Load context_encoder & candidate_encoder..'.format(device_index))
    context_encoder = Bert(preprocessor.spm_tokenizer.special_token_dict["pad"]["id"], **encoder_config["model"])
    context_encoder = load_state_dict(object=context_encoder, path=encoder_model_state_dict_path, map_location=device)
    candidate_encoder = Bert(preprocessor.spm_tokenizer.special_token_dict["pad"]["id"], **encoder_config["model"])
    candidate_encoder = load_state_dict(object=candidate_encoder, path=encoder_model_state_dict_path, map_location=device)

    # get arguments to pass 'main_worker' method
    print('From GPU Device {} ==> Setting model_params..'.format(device_index))
    model_params = PolyEncoder.get_init_params(context_encoder=context_encoder, candidate_encoder=candidate_encoder,
                                               m_code=model_config["m_code"], aggregation_method=model_config["aggregation_method"])

    print('From GPU Device {} ==> Setting arguments for main_worker..'.format(device_index))
    train_params = RetrieverFinetuningPolyEncoderTrainer.get_train_params(**train_config)
    data_params = RetrieverFinetuningPolyEncoderTrainer.get_data_params(train_data_dir=args.train_data_dir, val_data_dir=args.val_data_dir, encoding=data_config["encoding"], extension=data_config["extension"])

    # define trainer
    print('From GPU Device {} ==> Define trainer..'.format(device_index))
    trainer = RetrieverFinetuningPolyEncoderTrainer(temp_dir=train_config["temp_save_path"])
    trainer.set_ngpus_per_node(ngpus_per_node=nprocs)
    if train_config["lr_update"]:
        trainer.set_lr_update(initial_learning_rate=train_config["optimizer"]["initial_learning_rate"], num_warmup_steps=train_config["num_warmup_steps"])

    print('From GPU Device {} ==> Set train environments..'.format(device_index))
    local_batch_size = int(train_params["batch_size"])
    local_num_workers = trainer.num_workers
    local_pin_memory = trainer.pin_memory

    print("From GPU Device {} ==> Making model..".format(device_index))
    # define model, criterions, and optimizer
    model = trainer.create_model(**model_params)
    criterions, criterion_weights = trainer.get_criterions(**model_params, **train_params["criterion_weights"])
    optimizer = trainer.get_optimizer(model=model, **train_params["optimizer"])

    # load model.state_dict() when resume_path is not None
    if is_model_saved(path=train_params["resume_path"], save_hyperparms=False) and is_optimizer_saved(path=train_params["resume_path"], save_hyperparms=False):
        print("From GPU Device {} ==> Loading previous model from {}..".format(device_index, train_params["resume_path"]))
        model = load_state_dict(object=model, path=train_params["resume_path"], map_location=device)
        optimizer = load_state_dict(object=optimizer, path=train_params["resume_path"], map_location=device)
        if not trainer.lr_update:
            optimizer = trainer.update_optimizer_lr(optimizer=optimizer, lr=train_params["optimizer"]["initial_learning_rate"])
            print("From GPU Device {} ==> LearningRate has been set to {}..".format(device_index, train_params["optimizer"]["initial_learning_rate"]))

    # set device
    print("From GPU Device {} ==> Set gpu_device & DDP..".format(device_index))
    model = trainer.set_device(obj=model, device=device_index)
    optimizer = trainer.set_device(obj=optimizer, device=device_index)
    criterions = trainer.set_device(obj=criterions, device=device_index)

    # define data_loader
    print("From GPU Device {} ==> Preparing data_loader..".format(device_index))
    train_dataset = DatasetFromDir(data_dir=data_params["train_data_dir"], batch_size=local_batch_size, encoding=data_params["encoding"], extension=data_params["extension"], device=device, nprocs=nprocs)
    train_data_loader_params = trainer.get_data_loader_params(dataset=train_dataset, preprocessor=preprocessor, batch_size=local_batch_size,
                                                              device=device_index, nprocs=nprocs, num_workers=local_num_workers, pin_memory=local_pin_memory,
                                                              **model_params, **data_loader_config)
    train_data_loader = trainer.create_data_loader(**train_data_loader_params)

    val_data_loader = None
    if data_params["val_data_dir"] is not None:
        val_dataset = DatasetFromDir(data_dir=data_params["val_data_dir"], batch_size=local_batch_size, encoding=data_params["encoding"], extension=data_params["extension"], device=device_index, nprocs=nprocs)
        val_data_loader_params = trainer.get_data_loader_params(dataset=val_dataset, preprocessor=preprocessor, batch_size=local_batch_size,
                                                                device=device_index, nprocs=nprocs, num_workers=local_num_workers, pin_memory=local_pin_memory,
                                                                **model_params, **data_loader_config)
        val_data_loader = trainer.create_data_loader(**val_data_loader_params)

    # fit
    print("From GPU Device {} ==> Start training..".format(device_index))
    save_per_epoch = train_params["save_per_epoch"]
    save_per_batch = train_params["save_per_batch"]
    verbose_per_epoch = train_params["verbose_per_epoch"]
    verbose_per_batch = train_params["verbose_per_batch"]

    history = trainer.fit(model=model, train_data_loader=train_data_loader, val_data_loader=val_data_loader,
                          criterions=criterions, criterion_weights=criterion_weights, optimizer=optimizer,
                          device=device_index, epoch=train_params["epoch"], amp=train_params["amp"],
                          save_per_epoch=save_per_epoch, save_per_batch=save_per_batch, keep_last=train_params["keep_last"],
                          verbose_per_epoch=verbose_per_epoch, verbose_per_batch=verbose_per_batch)

    # save model.state_dict()
    save_path = train_params["save_path"] + "device_{}/".format(device_index)
    if not save_path.endswith("/"): save_path = save_path + "/"
    print("From GPU Device {} ==> Saving model.state_dict() into {}..".format(device_index, train_params["save_path"]))
    TrainerInterface.save(path=save_path, model=model, optimizer=optimizer, history=history, config=None,  preprocessor=preprocessor,
                          save_model_hyperparams=True, save_optimizer_hyperparams=False, ddp=False)
    preprocessor.save_spm_tokenizer(path=save_path)
    config["data"]["train_data_dir"] = args.train_data_dir
    config["data"]["val_data_dir"] = args.val_data_dir
    with open(save_path + ModelFilenameConstants.TRAIN_CONFIG_FILENAME, "w", encoding="utf-8") as fp:
        json.dump(config, fp)

if __name__ == '__main__':
    main()