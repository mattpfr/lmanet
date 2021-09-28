import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import numpy as np
import random

import models.erfnet as erfnet
import models.pspnet as pspnet
import utils.visualize

from datasets.helpers import DATASETS_NUM_CLASSES
from models.erfnet import UpsamplerBlock, non_bottleneck_1d
from models.correlation import FunctionCorrelation, FunctionCorrelationTranspose
from models.fusion import SimpleFusion, Fusion, FusionDown


class Encoder_M(nn.Module):
    def __init__(self, encoder):
        super(Encoder_M, self).__init__()
        self.encoder = encoder

    def forward(self, in_f):
        ret = self.encoder(in_f)
        return ret


class Encoder_Q(nn.Module):
    def __init__(self, encoder):
        super(Encoder_Q, self).__init__()
        self.encoder = encoder

    def forward(self, in_f):
        ret = self.encoder(in_f)
        return ret


class Decoder_STM(nn.Module):
    def __init__(self, num_classes, nobn, shallow_dec, input_dim):
        super().__init__()
        self.shallow_dec = shallow_dec
        if shallow_dec:
            self.output_conv_stm_shallow = nn.Conv2d(input_dim, num_classes, 1, stride=1, padding=0, bias=True)
        else:
            self.layers = nn.ModuleList()

            if input_dim == 128: # original ERFNet
                out_dim_1 = 64
                out_dim_2 = 16
            elif input_dim == 64: # modified for stm memory
                out_dim_1 = 32
                out_dim_2 = 16
            else:
                print("decoder for that input dim (",input_dim, ") Not implemented")
                exit(1)

            self.layers.append(UpsamplerBlock(input_dim, out_dim_1, nobn))
            self.layers.append(non_bottleneck_1d(out_dim_1, 0, 1, nobn))
            self.layers.append(non_bottleneck_1d(out_dim_1, 0, 1, nobn))

            self.layers.append(UpsamplerBlock(out_dim_1, out_dim_2, nobn))
            self.layers.append(non_bottleneck_1d(out_dim_2, 0, 1, nobn))
            self.layers.append(non_bottleneck_1d(out_dim_2, 0, 1, nobn))

            # The last convolution had != classes originally in cityscapes (20 instead of 24)
            self.output_conv = nn.ConvTranspose2d(out_dim_2, num_classes, 2, stride=2, padding=0, output_padding=0, bias=True)


    def forward(self, input):
        output = input
        if self.shallow_dec:
            output = self.output_conv_stm_shallow(output)
        else:
            for layer in self.layers:
                output = layer(output)

            output = self.output_conv(output)

        return output

    def set_to_eval_mode(self):
        self.eval()


class Memory(nn.Module):
    def __init__(self, args):
        super(Memory, self).__init__()
        self.learnable_constant = args.learnable_constant
        if self.learnable_constant:
            self.const = nn.Parameter(torch.zeros(1))

    def forward(self, m_in, m_out, q_in):  # m_in: o,c,t,h,w
        # o = batch of objects = num objects.
        # d is the dimension, number of channels, t is time
        B, D_e, T, H, W = m_in.size()
        _, D_o, _, _, _ = m_out.size()

        mi = m_in.view(B, D_e, T*H*W)
        mi = torch.transpose(mi, 1, 2)  # b, THW, emb

        qi = q_in.view(B, D_e, H*W)  # b, emb, HW

        p = torch.bmm(mi, qi) # b, THW, HW
        p = p / math.sqrt(D_e)
        if self.learnable_constant:
            p = torch.cat([p, self.const.view(1, 1, 1).expand(B, -1, H*W)], dim=1)
        p = F.softmax(p, dim=1) # b, THW, HW
        if self.learnable_constant:
            p = p[:, :-1, :]
        # For visualization later
        p_volume = None
        # p_volume = p.view(B, T, H, W, H, W)

        mo = m_out.view(B, D_o, T*H*W)

        mem = torch.bmm(mo, p) # Weighted-sum B, D_o, HW
        mem = mem.view(B, D_o, H, W)

        return mem, p, p_volume



class MemoryLocal(nn.Module):
    def __init__(self, args):
        super(MemoryLocal, self).__init__()
        self.learnable_constant = args.learnable_constant
        self.corr_size = args.corr_size
        if self.learnable_constant:
            self.const = nn.Parameter(torch.zeros(1))

    def forward(self, m_in, m_out, q_in):  # m_in: o,c,t,h,w
        # TODO note to verify
        # o = batch of objects = num objects. NOT batch
        # d is the dimension, number of channels, t is time
        B, D_e, T, H, W = m_in.size()
        _, D_o, _, _, _ = m_out.size()

        patch_size = self.corr_size

        p = torch.stack([FunctionCorrelation(q_in.contiguous(), m_in[:,:,t,:,:].contiguous(), patch_size) for t in range(T)], dim=2) # B, N^2, T, H, W
        p = p.reshape(B, -1, H, W) # B, T*N^2, H, W
        if self.learnable_constant:
            p = torch.cat([p, self.const.view(1, 1, 1, 1).expand(B, -1, H, W)], dim=1)
        p = F.softmax(p, dim=1)
        if self.learnable_constant:
            p = p[:, :-1, :, :]
        p = p.reshape(B, -1, T, H, W) # B, N^2, T, H, W

        p_volume = None
        # p_volume = torch.stack([self.remap_cost_volume(p[:,:,t,:,:]) for t in range(T)], dim=1)

        mem = sum([FunctionCorrelationTranspose(p[:,:,t,:,:].contiguous(), m_out[:,:,t,:,:].contiguous(), patch_size) for t in range(T)])

        return mem, p, p_volume

    def remap_cost_volume(self, cost_volume):
        """

        :param cost_volume: cost volume of shape (batch, (2*md-1)*(2*md-1), rows, cols), where md is the maximum displacement
                            allowed when computing the cost volume.
        :return: cost_volume_remapped: The input cost volume is remapped to shape (batch, rows, cols, rows, cols)
        """

        if cost_volume.dim() != 4:
            raise ValueError('input cost_volume should have 4 dimensions')

        [batch_size, d_, num_rows, num_cols] = cost_volume.size()
        d_sqrt_ = np.sqrt(d_)

        if not d_sqrt_.is_integer():
            raise ValueError("Invalid cost volume")

        cost_volume = cost_volume.view(batch_size, int(d_sqrt_), int(d_sqrt_), num_rows, num_cols)

        cost_volume_remapped = torch.zeros((batch_size, num_rows, num_cols,
                                            num_rows, num_cols),
                                           dtype=cost_volume.dtype,
                                           device=cost_volume.device)

        if cost_volume.size()[1] % 2 != 1:
            raise ValueError

        md = int((cost_volume.size()[1]-1)/2)

        for r in range(num_rows):
            for c in range(num_cols):
                r1_ = r - md
                r2_ = r1_ + 2*md + 1
                c1_ = c - md
                c2_ = c1_ + 2*md + 1

                r1_pad_ = max(-r1_, 0)
                r2_pad_ = max(r2_ - cost_volume_remapped.shape[1], 0)

                c1_pad_ = max(-c1_, 0)
                c2_pad_ = max(c2_ - cost_volume_remapped.shape[2], 0)

                d_ = cost_volume.size()[1]
                cost_volume_remapped[:, r1_+r1_pad_:r2_-r2_pad_, c1_+c1_pad_:c2_-c2_pad_, r, c] = \
                    cost_volume[:, r1_pad_:d_-r2_pad_, c1_pad_:d_-c2_pad_, r, c]

        return cost_volume_remapped


class MemoryQueue():

    def __init__(self, args):
        self.queue_size = args.stm_queue_size
        self.queue_keys = []
        self.queue_vals = []
        self.queue_idxs = []

    def reset(self):
        self.queue_keys = []
        self.queue_vals = []
        self.queue_idxs = []

    def current_size(self):
        return len(self.queue_keys)

    def update(self, key, val, idx):
        self.queue_keys.append(key)
        self.queue_vals.append(val)
        self.queue_idxs.append(idx)

        if len(self.queue_keys) > self.queue_size:
            self.queue_keys.pop(0)
            self.queue_vals.pop(0)
            self.queue_idxs.pop(0)

    def get_indices(self):
        return self.queue_idxs

    def get_keys(self):
        return torch.stack(self.queue_keys, dim=2)

    def get_vals(self):
        return torch.stack(self.queue_vals, dim=2)


class KeyValue(nn.Module):
    # Not using location
    def __init__(self, indim, keydim, valdim, val_pass):
        super(KeyValue, self).__init__()
        self.Key = nn.Conv2d(indim, keydim, kernel_size=(3,3), padding=(1,1), stride=1)
        self.val_pass = val_pass
        if not self.val_pass:
            self.Value = nn.Conv2d(indim, valdim, kernel_size=(3,3), padding=(1,1), stride=1)

    def forward(self, x):
        val = x if self.val_pass else self.Value(x)
        return self.Key(x), val


class LMANet(nn.Module):
    def __init__(self, args, encoder=None, nobn=False, print_all_logs=True, board=None):  # use encoder to pass pretrained encoder
        super(LMANet, self).__init__()
        self.num_classes = DATASETS_NUM_CLASSES[args.dataset]
        self.is_training = args.training
        self.shallow_dec = args.shallow_dec
        self.model_struct = args.model_struct
        self.align_weights = args.align_weights
        self.baseline_mode = args.baseline_mode
        self.local_correlation = args.local_correlation
        self.always_decode = args.always_decode
        self.memorize_first = args.memorize_first
        self.memory_strategy = args.memory_strategy
        self.stm_queue_size = args.stm_queue_size
        self.backbone = args.backbone
        self.backbone_nobn = args.backbone_nobn
        self.board = board

        assert(not(self.model_struct == "original_kv_fusion" and self.memorize_first))

        if encoder:
            self.encoder = encoder
            print("Case not supported")
            exit(1)

        # Created here, but the model weights are loaded afterwards from file
        if args.backbone == "erf":
            self.encoder = erfnet.Encoder(args.training, self.backbone_nobn)
            fea_dim = 128
            stm_val_dim = 64
            stm_key_dim = 16
            val_pass = False
        elif args.backbone == "psp101":
            self.encoder = pspnet.PSPEncoder(pretrained=True, layers=101, nobn=self.backbone_nobn)
            fea_dim = 2048
            stm_val_dim = 2048 # For psp, we use directly the encoder features as Value
            stm_key_dim = 128
            val_pass = True
        else:
            print("Unknown backbone")
            exit(1)

        self.kv_M_r4 = KeyValue(fea_dim, keydim=stm_key_dim, valdim=stm_val_dim, val_pass=val_pass)
        self.kv_Q_r4 = KeyValue(fea_dim, keydim=stm_key_dim, valdim=stm_val_dim, val_pass=val_pass)
        if self.model_struct == "original_kv_fusion":
            self.kv_init_r4 = KeyValue(fea_dim, keydim=stm_key_dim, valdim=stm_val_dim, val_pass=val_pass)

        # Aligning stm weights at init for training to a unique value
        if self.align_weights:
            self.kv_Q_r4.Key.weight.data = self.kv_M_r4.Key.weight.data
            self.kv_Q_r4.Key.bias.data = self.kv_M_r4.Key.bias.data

        if self.local_correlation:
            self.memory_module_local = MemoryLocal(args)
        else:
            self.memory_module = Memory(args)

        if args.fusion_strategy == "simple-sigmoid":
            self.memory_fusion = SimpleFusion(args, nobn)
        elif args.fusion_strategy in ["sigmoid-down1"]:
            self.memory_fusion = FusionDown(args, stm_val_dim, nobn)
        else:
            self.memory_fusion = Fusion(args, stm_val_dim, nobn)
        self.memory_queue = MemoryQueue(args)

        # Decode features
        if args.backbone == "erf":
            # self.decoder = erfnet.Decoder(self.num_classes, nobn, self.shallow_dec)
            self.decoder = Decoder_STM(self.num_classes, self.backbone_nobn, self.shallow_dec, 2*stm_val_dim)
            if self.model_struct == "original" or self.model_struct == "original_kv_fusion":
                # Nothing else to do
                pass
            elif self.model_struct == "mem_only" or self.model_struct == "aux_mem_loss":
                self.decoder_mem = Decoder_STM(self.num_classes, self.backbone_nobn, self.shallow_dec, stm_val_dim)
            else:
                print("model_struct Not implemented")
                exit(1)
        elif args.backbone == "psp101":
            self.use_ppm = True
            self.zoom_factor = 8
            bins = (1, 2, 3, 6)
            dropout = 0.1
            classes = 19
            zoom_factor = 8
            assert(2048 % len(bins) == 0)
            assert(self.zoom_factor in [1, 2, 4, 8])
            if self.use_ppm:
                self.ppm = pspnet.PPM(fea_dim, int(fea_dim/len(bins)), bins)
                fea_dim *= 2

            self.decoder = pspnet.PSPDecoder(zoom_factor, fea_dim, dropout, classes, nobn)
            if self.model_struct == "original" or self.model_struct == "original_kv_fusion":
                # Nothing else to do
                pass
            elif self.model_struct == "mem_only" or self.model_struct == "aux_mem_loss":
                self.decoder_mem = pspnet.PSPDecoder(zoom_factor, fea_dim, dropout, classes)
            else:
                print("model_struct Not implemented")
                exit(1)
        else:
            print("Unknown backbone")
            exit(1)

        if print_all_logs:
            utils.visualize.print_summary(self.__dict__, self.__class__.__name__)

    def reset_hidden_state(self):
        self.memory_queue.reset()

    def memory_range(self, seq_len):

        ret_range = range(seq_len)

        if self.memory_strategy == "all":
            pass
        if self.memory_strategy == "skip_01":
            ret_range = range(0, seq_len + 1, 2)
        if self.memory_strategy == "skip_02":
            ret_range = range(0, seq_len + 1, 3)
        if self.memory_strategy == "skip_03":
            ret_range = range(0, seq_len + 1, 4)
        if self.memory_strategy == "skip_04":
            ret_range = range(0, seq_len + 1, 5)
        if self.memory_strategy == "skip_05":
            ret_range = range(0, seq_len + 1, 6)
        if self.memory_strategy == "random":
            ret_range = random.sample(range(seq_len-1), self.stm_queue_size-1)
            ret_range.append(seq_len - 1)
            ret_range.sort()

        # assert(len(ret_range) == self.stm_queue_size)

        assert(seq_len - 1 in ret_range)
        return ret_range

    def forward(self, input, step=0, epoch=0):

        # Input has to be of size (batch_size, seq_len, channels, h, w)
        seq_len = input.size(1)

        # If seq_len == 1, are evaluating, otherwise we are probably in training mode

        decoder_outputs = []
        decoder_outputs_aux = []
        memory_range = self.memory_range(seq_len)
        for t in memory_range:

            if self.backbone == "psp101":
                input_size = input[:, t, :, :, :].size()
                assert((input_size[2]-1) % 8 == 0 and (input_size[3]-1) % 8 == 0)

                h = int((input_size[2] - 1) / 8 * self.zoom_factor + 1)
                w = int((input_size[3] - 1) / 8 * self.zoom_factor + 1)

            encoder_output = self.encoder(input[:, t, :, :, :])

            if self.baseline_mode:

                if not (self.always_decode or (t == seq_len - 1)):
                    continue

                if self.backbone == "erf":
                    output_decoded = self.decoder.forward(encoder_output)
                elif self.backbone == "psp101":
                    output_decoded = self.ppm(encoder_output)
                    output_decoded, _ = self.decoder(output_decoded, h, w)
                decoder_outputs.append(output_decoded)
                continue

            if self.memorize_first:
                kM, vM = self.kv_M_r4.forward(encoder_output)

                idx = step if seq_len == 1 else t
                self.memory_queue.update(kM, vM, idx)

            # H/16 and W/16 for 384 => 24 x 24 for the encoder output
            # Here HxW is 64x128 for the encoder output...
            if self.always_decode or (t == seq_len - 1) or self.model_struct == "original_kv_fusion":
                if self.memory_queue.current_size() != 0:
                    # assert(t != 0)
                    kQ, vQ = self.kv_Q_r4.forward(encoder_output)

                    # TODO original comment: memory select kv:(1, K, C, T, H, W) ???
                    if self.local_correlation:
                        mem, p, p_vol = self.memory_module_local.forward(
                            self.memory_queue.get_keys(), self.memory_queue.get_vals(), kQ)
                    else:
                        mem, p, p_vol = self.memory_module.forward(
                            self.memory_queue.get_keys(), self.memory_queue.get_vals(), kQ)

                    fused_mem = self.memory_fusion(mem, vQ)

                    if self.model_struct == "original_kv_fusion":
                        kM, vM = self.kv_M_r4.forward(fused_mem)
                        idx = step if seq_len == 1 else t
                        self.memory_queue.update(kM, vM, idx)

                    if not (self.always_decode or (t == seq_len - 1)):
                        continue

                    # To compare the two (visualization of correlation maps side by side)
                    #mem_l, mem_concat_l, p_l, p_vol_l = self.memory_module_local.forward(self.memory_queue.get_keys(), self.memory_queue.get_vals(), kQ, vQ)

                    if self.model_struct == "original" or self.model_struct == "original_kv_fusion":
                        if self.backbone == "erf":
                            output_decoded = self.decoder.forward(fused_mem)
                        elif self.backbone == "psp101":
                            output_decoded = self.ppm(fused_mem)
                            output_decoded, _ = self.decoder(output_decoded, h, w)
                        decoder_outputs.append(output_decoded)
                    elif self.model_struct == "mem_only":
                        if self.backbone == "erf":
                            output_decoded = self.decoder.forward(mem)
                        elif self.backbone == "psp101":
                            output_decoded = self.ppm(mem)
                            output_decoded, _ = self.decoder(output_decoded, h, w)
                        decoder_outputs.append(output_decoded_mem)
                    elif self.model_struct == "aux_mem_loss":
                        if self.backbone == "erf":
                            output_decoded = self.decoder.forward(fused_mem)
                        elif self.backbone == "psp101":
                            output_decoded = self.ppm(fused_mem)
                            output_decoded, _ = self.decoder(output_decoded, h, w)
                        decoder_outputs.append(output_decoded)

                        if self.backbone == "erf":
                            output_decoded_mem = self.decoder_mem.forward(mem)
                        elif self.backbone == "psp101":
                            output_decoded_mem = self.ppm(mem)
                            output_decoded_mem, _ = self.decoder_mem(output_decoded_mem, h, w)
                        decoder_outputs_aux.append(output_decoded_mem)
                else:
                    # First case without memory, simply predict
                    # torch.Size([4, 20, 64, 128])
                    # torch.Size([4, 20, 512, 1024])

                    if self.model_struct == "original_kv_fusion":
                        kInit, vInit = self.kv_init_r4.forward(encoder_output)
                        idx = step if seq_len == 1 else t
                        self.memory_queue.update(kInit, vInit, idx)

                    if not (self.always_decode or (t == seq_len - 1)):
                        continue

                    if self.backbone == "erf":
                        output_decoded = self.decoder.forward(encoder_output)
                    elif self.backbone == "psp101":
                        output_decoded = self.ppm(encoder_output)
                        output_decoded, _ = self.decoder(output_decoded, h, w)

                    decoder_outputs.append(output_decoded)
                    if self.model_struct == "aux_mem_loss":
                        decoder_outputs_aux.append(output_decoded)

            # Memorize
            if not self.memorize_first and self.model_struct != "original_kv_fusion":
                kM, vM = self.kv_M_r4.forward(encoder_output)

                idx = step if seq_len == 1 else t
                self.memory_queue.update(kM, vM, idx)

        output_decoder = torch.stack(decoder_outputs, dim=1)
        output_decoder_aux = None
        if decoder_outputs_aux:
            output_decoder_aux = torch.stack(decoder_outputs_aux, dim=1)

        return output_decoder, output_decoder_aux
