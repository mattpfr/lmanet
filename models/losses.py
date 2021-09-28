import torch
from datasets.helpers import classes_weights, DATASETS_CLASSES_DICT

import torch.nn.functional as F


class CrossEntropyLoss2d(torch.nn.Module):

    def __init__(self, args, gpu):
        super().__init__()

        weights = classes_weights(DATASETS_CLASSES_DICT[args.dataset], args.shallow_dec, gpu)

        self.win_size = args.win_size
        self.always_decode = args.always_decode

        if args.dataset == "Cityscapes":
            n_losses = 1
            self.last_frame_loss = True
            self.idx_start = self.win_size - 1
        else:
            n_losses = -1
            print("unsupported dataset")
            exit(1)

        self.loss_idx_step = 1
        if args.model_struct == "aux_mem_loss":
            n_losses *= 2
            self.loss_idx_step = 2

        self.losses = [torch.nn.NLLLoss(weights, ignore_index=19) for i in range(n_losses)]

        assert(len(self.losses) == n_losses)
        assert(self.idx_start < self.win_size)

    def forward(self, probs, labels, probs_aux):
        # Inputs are of size (batch_size, seq_len, channels, h, w)

        if self.always_decode:
            assert(probs.size(1) == self.win_size)
        else:
            assert(probs.size(1) == 1)

        losses = []

        softmax_dim = 1

        for t in range(self.idx_start, self.win_size, self.loss_idx_step):
            label_idx = 0 if self.last_frame_loss else t
            probs_idx = 0 if not self.always_decode else t
            losses.append(self.losses[t - self.idx_start]
                          (F.log_softmax(probs[:,probs_idx,:,:,:], dim=softmax_dim), labels[:,label_idx,0,:,:]))
            if probs_aux is not None:
                losses.append(self.losses[t - self.idx_start + 1]
                              (F.log_softmax(probs_aux[:,probs_idx,:,:,:], dim=softmax_dim), labels[:,label_idx,0,:,:]))

        return sum(losses) / len(losses)