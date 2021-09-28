import torch
import torch.nn as nn
import torch.nn.functional as F

import utils.visualize

class DownsamplerBlock (nn.Module):
    def __init__(self, ninput, noutput, nobn):
        super().__init__()

        self.nobn = nobn
        self.conv = nn.Conv2d(ninput, noutput-ninput, (3, 3), stride=2, padding=1, bias=True)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):
        output = torch.cat([self.conv(input), self.pool(input)], 1)
        if not self.nobn:
            output = self.bn(output)
        return F.relu(output, inplace=self.nobn)
    

class non_bottleneck_1d (nn.Module):
    def __init__(self, chann, dropprob, dilated, nobn):
        super().__init__()

        self.nobn = nobn

        self.conv3x1_1 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(1, 0), bias=True)

        self.conv1x3_1 = nn.Conv2d(chann, chann, (1, 3), stride=1, padding=(0, 1), bias=True)

        self.bn1 = nn.BatchNorm2d(chann, eps=1e-03)

        self.conv3x1_2 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(1 * dilated, 0), bias=True, dilation=(dilated, 1))

        self.conv1x3_2 = nn.Conv2d(chann, chann, (1, 3), stride=1, padding=(0, 1 * dilated), bias=True, dilation=(1, dilated))

        self.bn2 = nn.BatchNorm2d(chann, eps=1e-03)

        self.dropout = nn.Dropout2d(dropprob)
        

    def forward(self, input):

        output = self.conv3x1_1(input)
        output = F.relu(output, inplace=self.nobn)
        output = self.conv1x3_1(output)
        if not self.nobn:
            output = self.bn1(output)
        output = F.relu(output, inplace=self.nobn)

        output = self.conv3x1_2(output)
        output = F.relu(output)
        output = self.conv1x3_2(output)
        if not self.nobn:
            output = self.bn2(output)
            if (self.dropout.p != 0):
                output = self.dropout(output)
        
        return F.relu(output+input, inplace=self.nobn)    #+input = identity (residual connection)


class Encoder(nn.Module):
    def __init__(self, is_training, nobn):
        super().__init__()
        self.initial_block = DownsamplerBlock(3, 16, nobn)

        self.layers = nn.ModuleList()

        self.layers.append(DownsamplerBlock(16, 64, nobn))

        drop_prob = 0.03 if is_training else 0.1
        for x in range(0, 5):    #5 times
           self.layers.append(non_bottleneck_1d(64, drop_prob, 1, nobn))   #Dropout here was wrong in prev trainings

        self.layers.append(DownsamplerBlock(64, 128, nobn))

        drop_prob = 0.3 if is_training else 0.1
        for x in range(0, 2):    #2 times
            self.layers.append(non_bottleneck_1d(128, drop_prob, 2, nobn))
            self.layers.append(non_bottleneck_1d(128, drop_prob, 4, nobn))
            self.layers.append(non_bottleneck_1d(128, drop_prob, 8, nobn))
            self.layers.append(non_bottleneck_1d(128, drop_prob, 16, nobn))

        # only for encoder mode:
        # The last convolution had != classes originally in cityscapes (20 instead of 24)
        # self.output_conv = nn.Conv2d(128, num_classes, 1, stride=1, padding=0, bias=True)

    def forward(self, input):
        output = self.initial_block(input)

        for layer in self.layers:
            output = layer(output)

        return output

    def set_to_eval_mode(self):
        self.eval()


class UpsamplerBlock (nn.Module):
    def __init__(self, ninput, noutput, nobn):
        super().__init__()
        self.nobn = nobn
        self.conv = nn.ConvTranspose2d(ninput, noutput, 3, stride=2, padding=1, output_padding=1, bias=True)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):
        output = self.conv(input)
        if not self.nobn:
            output = self.bn(output)
        return F.relu(output, inplace=self.nobn)

class Decoder (nn.Module):
    def __init__(self, num_classes, nobn, shallow_dec):
        super().__init__()
        self.shallow_dec = shallow_dec
        if shallow_dec:
            self.output_conv_new_shallow = nn.Conv2d(128, num_classes, 1, stride=1, padding=0, bias=True)
        else:
            self.layers = nn.ModuleList()

            self.layers.append(UpsamplerBlock(128, 64, nobn))
            self.layers.append(non_bottleneck_1d(64, 0, 1, nobn))
            self.layers.append(non_bottleneck_1d(64, 0, 1, nobn))

            self.layers.append(UpsamplerBlock(64, 16, nobn))
            self.layers.append(non_bottleneck_1d(16, 0, 1, nobn))
            self.layers.append(non_bottleneck_1d(16, 0, 1, nobn))

            self.output_conv_new = nn.ConvTranspose2d(16, num_classes, 2, stride=2, padding=0, output_padding=0, bias=True)

    def forward(self, input):
        output = input
        if self.shallow_dec:
            output = self.output_conv_new_shallow(output)
        else:
            for layer in self.layers:
                output = layer(output)

            output = self.output_conv_new(output)

        return output


class ERFNet(nn.Module):
    def __init__(self, num_classes, is_training, shallow_dec, encoder=None, nobn=False, print_all_logs=True):  #use encoder to pass pretrained encoder
        super().__init__()
        self.num_classes = num_classes
        self.is_training = is_training
        if not encoder:
            self.encoder = Encoder(is_training, nobn)
        else:
            self.encoder = encoder
        self.decoder = Decoder(num_classes, nobn, shallow_dec)
        if print_all_logs:
            utils.visualize.print_summary(self.__dict__, self.__class__.__name__)

    def forward(self, input):
        encoder_output = self.encoder.forward(input)
        return self.decoder.forward(encoder_output)