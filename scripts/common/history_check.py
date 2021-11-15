import os
import sys
import json
import pickle
import numpy as np
import argparse
sys.path.append(os.path.abspath('./'))

repr_template = " || {mode}({iteration:^10d}) | {loss_repr} | {acc_repr}"
lr_template = " | last_lr: {last_lr:.5e}"
loss_template = "{key}_loss: {value:.5e}, "
acc_template = "{key}_acc: {value:.5e}, "

def parse():
    parser = argparse.ArgumentParser(description="Check training history")
    parser.add_argument("--history_path", metavar="History pickle file path", help="specify history pickle path; e.g. './model/temp/20210803_145711/history.pickle")
    parser.add_argument("--epoch", metavar="Epoch", help="total epoch")
    args = parser.parse_args()
    return args

def get_repr(history, begin_idx, end_idx, mode="train"):
    loss_repr = ""
    for k, v in history[mode]["loss"].items():
        if k.startswith("total"): continue
        value = np.mean(v[begin_idx:end_idx])
        loss_repr += loss_template.format(key=k, value=value)
    acc_repr = ""
    for k, v in history[mode]["acc"].items():
        if k.startswith("total"): continue
        value = np.mean(v[begin_idx:end_idx])
        acc_repr += acc_template.format(key=k, value=value)
    return loss_repr, acc_repr

def main():
    # parse arguments
    args = parse()
    history = None
    with open(args.history_path, "rb") as fp:
        history = pickle.load(fp)
    epoch = int(args.epoch)

    train_iter_size = int(history["train"]["iteration"])
    train_batch_size = train_iter_size // epoch

    val_iter_size = None
    val_batch_size = None
    if history["val"]["iteration"] > 0:
        val_iter_size = int(history["train"]["iteration"])
        val_batch_size = val_iter_size // epoch

    for e in range(1, epoch+1):
        repr = "Epoch {epoch:3d}".format(epoch=e)

        train_begin_idx = train_batch_size * (e - 1)
        train_end_idx = train_batch_size * e
        train_loss_repr, train_acc_repr = get_repr(history=history, begin_idx=train_begin_idx, end_idx=train_end_idx, mode="train")
        repr += repr_template.format(mode="train", iteration=(train_end_idx-train_begin_idx), loss_repr=train_loss_repr[:-2], acc_repr=train_acc_repr[:-2])
        repr += lr_template.format(last_lr=history["train"]["lr"][train_end_idx-1])

        if val_batch_size is not None:
            val_begin_idx = val_batch_size * (e - 1)
            val_end_idx = val_batch_size * e
            val_loss_repr, val_acc_repr = get_repr(history=history, begin_idx=val_begin_idx, end_idx=val_end_idx, mode="val")
            repr += repr_template.format(mode="val", iteration=(val_end_idx-val_begin_idx), loss_repr=val_loss_repr[:-2], acc_repr=val_acc_repr[:-2])

        print(repr)

if __name__ == '__main__':
    main()