from train import train


def main(model_config=None):
    modelConfig = {
        "state": "train",  # or eval
        "epoch": 50,
        "batch_size": 32,
        "n_steps": 1000,
        "n_channel": 64,
        "ch_mult": [1, 2, 2, 4],
        "has_attn": [False, False, True, True],
        "n_block": 2,
        "lr": 1e-4,
        "img_size": 32,
        "grad_clip": 1.,
        "device": "cuda:0",
        "training_load_weight": None,
        "save_weight_dir": "./Checkpoints/",
    }
    if model_config is not None:
        modelConfig = model_config
    if modelConfig["state"] == "train":
        train(modelConfig)
    else:
        eval(modelConfig)


if __name__ == '__main__':
    main()