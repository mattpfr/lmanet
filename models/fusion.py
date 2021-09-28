import torch
import torch.nn as nn

import torch.nn.functional as F


class Fusion(nn.Module):
    def __init__(self, args, stm_val_dim, nobn):
        super(Fusion, self).__init__()

        self.nobn = nobn
        self.fusion_strategy = args.fusion_strategy

        if self.fusion_strategy in ["sigmoid-do1", "sigmoid-do2", "sigmoid-do3", "concat"]:

            dropprob = 0.3 if self.fusion_strategy in ["sigmoid-do1", "concat", "sigmoid-do3"] else 0.0
            self.dropout1 = nn.Dropout2d(dropprob)
            self.dropout2 = nn.Dropout2d(dropprob)
            self.bn1 = nn.BatchNorm2d(stm_val_dim, eps=1e-03)
            self.bn2 = nn.BatchNorm2d(stm_val_dim, eps=1e-03)

        if self.fusion_strategy != "concat":
            self.conv_layer_1 = nn.Conv2d(stm_val_dim,  2*stm_val_dim, (3, 3), stride=1, padding=1, bias=True)
            self.conv_layer_2 = nn.Conv2d(stm_val_dim,  2*stm_val_dim, (3, 3), stride=1, padding=1, bias=True)
            if self.fusion_strategy == "sigmoid-do3":
                self.conv_layer_34 = nn.Conv2d(2*stm_val_dim, 2*stm_val_dim, (3, 3), stride=1, padding=1, bias=True)
            else:
                self.conv_layer_3 = nn.Conv2d(2*stm_val_dim, 2*stm_val_dim, (3, 3), stride=1, padding=1, bias=True)
                self.conv_layer_4 = nn.Conv2d(2*stm_val_dim, 2*stm_val_dim, (3, 3), stride=1, padding=1, bias=True)

            self.bn = nn.BatchNorm2d(2*stm_val_dim, eps=1e-03)

    def forward(self, mem, vQ):

        in_1 = vQ
        in_2 = mem

        if self.fusion_strategy in ["sigmoid-do1", "sigmoid-do2", "sigmoid-do3", "concat"]:

            in_1 = self.bn1(in_1)
            in_1 = self.dropout1(in_1)
            in_1 = F.relu(in_1, inplace=False)

            in_2 = self.bn2(in_2)
            in_2 = self.dropout2(in_2)
            in_2 = F.relu(in_2, inplace=False)

        fused_mem = torch.cat([in_1, in_2], dim=1)

        if self.fusion_strategy != "concat":
            if self.fusion_strategy == "sigmoid-do3":
                att = self.conv_layer_34(fused_mem)
            else:
                att_1 = self.conv_layer_3(fused_mem)
                att_2 = self.conv_layer_4(fused_mem)

            out_1 = self.conv_layer_1(in_1)
            out_2 = self.conv_layer_2(in_2)

            if self.fusion_strategy in ["sigmoid", "sigmoid-do1", "sigmoid-do2"]:
                out_1 *= torch.sigmoid(att_1)
                out_2 *= torch.sigmoid(att_2)
            elif self.fusion_strategy == "sigmoid-do3":
                out_1 *= torch.sigmoid(att)
                out_2 *= torch.sigmoid(att)
            else:
                out_1 *= att_1
                out_2 *= att_2

            fused_mem = out_1 + out_2
            if not self.nobn:
                fused_mem = self.bn(fused_mem)
            fused_mem = F.relu(fused_mem, inplace=self.nobn)

        return fused_mem


class FusionDown(nn.Module):
    def __init__(self, args, stm_val_dim, nobn):
        super(FusionDown, self).__init__()

        self.nobn = nobn
        self.fusion_strategy = args.fusion_strategy

        down_dim = 256
        # dropprob = 0.3 if self.fusion_strategy == "sigmoid-down1" else 0.0

        # self.dropout1 = nn.Dropout2d(dropprob)
        # self.dropout2 = nn.Dropout2d(dropprob)
        self.bn0 = nn.BatchNorm2d(down_dim, eps=1e-03)

        ks = 1
        pad = 0
        self.conv_layer_1 = nn.Conv2d(stm_val_dim, down_dim, (ks, ks), stride=1, padding=pad, bias=True)
        self.conv_layer_2 = nn.Conv2d(stm_val_dim, down_dim, (ks, ks), stride=1, padding=pad, bias=True)
        self.conv_layer_3 = nn.Conv2d(down_dim, stm_val_dim, (ks, ks), stride=1, padding=pad, bias=True)
        self.conv_layer_4 = nn.Conv2d(down_dim, stm_val_dim, (ks, ks), stride=1, padding=pad, bias=True)

        self.bn = nn.BatchNorm2d(stm_val_dim, eps=1e-03)

    def forward(self, mem, vQ):

        in_1 = vQ
        in_2 = mem

        # Directly using resnet features, that already have bn, no drop
        down_1 = self.conv_layer_1(in_1)

        # TODO do we need bn/drop/relu before going down in dimension from the memory
        down_2 = self.conv_layer_1(in_2)

        fused_mem = down_1 + down_2
        if not self.nobn:
            fused_mem = self.bn0(fused_mem)
        fused_mem = F.relu(fused_mem, inplace=self.nobn)

        att_1 = self.conv_layer_3(fused_mem)
        att_2 = self.conv_layer_4(fused_mem)

        out_1 = in_1 * torch.sigmoid(att_1)
        out_2 = in_2 * torch.sigmoid(att_2)

        fused_mem = out_1 + out_2
        if not self.nobn:
            fused_mem = self.bn(fused_mem)
        fused_mem = F.relu(fused_mem, inplace=self.nobn)

        return fused_mem



class SimpleFusion(nn.Module):
    def __init__(self, args, nobn):
        super(SimpleFusion, self).__init__()

        self.nobn = nobn
        self.fusion_strategy = args.fusion_strategy

        if self.fusion_strategy != "concat":
            self.conv_layer_1 = nn.Conv2d(64,  128, (3, 3), stride=1, padding=1, bias=True)
            self.conv_layer_2 = nn.Conv2d(64,  128, (3, 3), stride=1, padding=1, bias=True)
            self.conv_layer_3 = nn.Conv2d(128, 128, (3, 3), stride=1, padding=1, bias=True)

            self.bn = nn.BatchNorm2d(128, eps=1e-03)

    def forward(self, mem, vQ):

        in_1 = vQ
        in_2 = mem


        fused_mem = torch.cat([in_1, in_2], dim=1)

        if self.fusion_strategy != "concat":
            att_1 = self.conv_layer_3(fused_mem)

            out_1 = self.conv_layer_1(in_1)
            out_2 = self.conv_layer_2(in_2)

            if self.fusion_strategy in ["simple-sigmoid"]:
                out_1 *= torch.sigmoid(att_1)
                out_2 *= torch.sigmoid(att_1)
            else:
                out_1 *= att_1
                out_2 *= att_1

            fused_mem = out_1 + out_2
            if not self.nobn:
                fused_mem = self.bn(fused_mem)
            fused_mem = F.relu(fused_mem, inplace=self.nobn)

        return fused_mem