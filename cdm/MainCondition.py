from TrainCondition import train, eval

# 修改适配于数据集的图像参数
def main(model_config=None):
    modelConfig = {
        "label" : 3 ,
        "state": "train", # or eval
        "epoch": 500,
        "batch_size": 512,
        "T": 1000,
        "channel": 128,
        "channel_mult": [1, 2, 2, 2],
        "num_res_blocks": 2,
        "dropout": 0.15,
        "lr": 1e-4,
        "multiplier": 2.5,
        "beta_1": 1e-4,
        "beta_T": 0.028,
        "img_size": 28,
        "grad_clip": 1.,
        "device": "cuda:0",
        "w": 1.8,
        "save_dir": "",
        "training_load_weight": None,
        "test_load_weight": "ckpt_99_.pt",
        "sampled_dir": "",
        "sampledNoisyImgName": ".png",
        "sampledImgName": ".png",
        "nrow": 8
    }
    if model_config is not None:
        modelConfig = model_config
    if modelConfig["state"] == "train":
        train(modelConfig)
    else:
        eval(modelConfig)


if __name__ == '__main__':
    main()
