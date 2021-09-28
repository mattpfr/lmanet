import torch
import re
import os


from models.erfnet import ERFNet
from models.lmanet import LMANet

models = {
    "erfnet": ERFNet,
    "lmanet": LMANet
}

def load_state_dict(model, state_dict, try_load_module_name=True, print_all_logs=True, backbone="erf"):
    own_state = model.state_dict()
    load_decoder_aux_params = False # might be useful later
    total_loaded = 0
    total_skipped = 0

    if backbone == "erf":
        state_dict_items = state_dict.items()
    elif backbone == "psp101":
        state_dict_items = state_dict.items()
    else:
        print("unknown backbone")
        exit(1)

    for name, param in state_dict_items:
        if name not in own_state:
            if try_load_module_name and name.startswith("module."):
                own_state[name.split("module.")[-1]].copy_(param)
                total_loaded += 1
            elif backbone == "psp101" and name.replace("module.layer", "module.encoder.layer") in own_state:
                own_state[name.replace("module.layer", "module.encoder.layer")].copy_(param)
                total_loaded += 1
                # print("replaced", name, "by", name.replace("module.layer", "module.encoder.layer"))
            elif backbone == "psp101" and name.replace("module.cls", "module.decoder.cls") in own_state:
                own_state[name.replace("module.cls", "module.decoder.cls")].copy_(param)
                total_loaded += 1
                # print("replaced", name, "by", name.replace("module.cls", "module.decoder.cls"))
            else:
                if load_decoder_aux_params:
                    dec_aux_name = name.replace("decoder", "decoder_aux")
                    own_state[dec_aux_name].copy_(param)
                    total_loaded += 1
                    if print_all_logs:
                        print("WARN: ", dec_aux_name, "loaded from", name)
                else:
                    if print_all_logs:
                        print(name, " not loaded")
                    total_skipped += 1
                    continue
        else:
            own_state[name].copy_(param)
            total_loaded += 1

    if print_all_logs:
        print("Total params loaded :", total_loaded)
        print("Total params skipped:", total_skipped)

    return model


def load_model(model, weights_path):

    print("Loading weights for model", model._get_name())
    if not weights_path:
        print("No weights file specified, not loading paramaters...")
        return model
    elif os.path.exists(weights_path):
        print("Loading weights from: ", weights_path)
    else:
        print("Could not load weights, file does not exist: ", weights_path)
        return model

    model = torch.nn.DataParallel(model).cuda()

    model = load_state_dict(model, torch.load(weights_path,
                                              map_location=lambda storage, loc: storage))
    print("Weights loaded successfully!")

    return model


def load_model_from_file(args, model_path, board, gpu, checkpoint):

    if not os.path.exists(model_path):
        print("Could not load model, file does not exist: ", model_path)
        exit(1)

    print_all_logs = gpu == 0

    model_name = str(os.path.basename(model_path).split('.')[0])
    model = models[model_name](args, print_all_logs=print_all_logs, board=board)
    if print_all_logs:
        print("Loaded model file: ", model_path)

    torch.cuda.set_device(gpu)
    model.cuda(gpu)

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu], find_unused_parameters=True)
    if args.gpus > 1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    weights_path = args.weights
    if weights_path:
        if os.path.exists(weights_path):
            if print_all_logs:
                print("Loading weights file: ", weights_path)
        else:
            print("Could not load weights, file does not exist: ", weights_path)

        if checkpoint is not None:
            weights_dict = checkpoint['state_dict']
        else:
            weights_dict = torch.load(weights_path, map_location=lambda storage, loc: storage)

        # print(torch.load(args.state))
        model = load_state_dict(model,
                                weights_dict,
                                try_load_module_name=False, print_all_logs=print_all_logs,
                                backbone=args.backbone)
        if print_all_logs:
            print("Loaded weights:", weights_path)

    filter_model_params_optimization(args, model)

    return model



def load_checkpoint(save_dir, enc):
    tag = "_enc" if enc else ""
    checkpoint_path = os.path.join(save_dir, "checkpoint{}.pth.tar".format(tag))

    if os.path.exists(checkpoint_path):
        print("Loading checkpoint file: ", checkpoint_path)
    else:
        print("Could not load checkpoint, file does not exist: ", checkpoint_path)

    return torch.load(checkpoint_path, map_location=lambda storage, loc: storage)


def filter_model_params_optimization(args, model):

    if not args.training:
        for name, param in model.named_parameters():
            param.requires_grad = False
    else:
        for name, param in model.named_parameters():
            if re.match("module.encoder.*", name):
                param.requires_grad = args.train_encoder
            if re.match("module.decoder.*", name) or re.match("module.ppm.*", name):
                param.requires_grad = args.train_decoder
            if re.match(".*output_conv_new.*", name):
                param.requires_grad = args.train_erfnet_shallow_dec
            if re.match(".*output_conv_stm_shallow.*", name):
                param.requires_grad = True