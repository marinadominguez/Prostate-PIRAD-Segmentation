"""
3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation
Paper URL: https://arxiv.org/abs/1606.06650
Author: Amir Aghdam
"""
import sys
import os

# Add the project root to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from torch import nn
from torchsummary import summary
import torch
import time
import HOSP_pipeline.config.config as cfg

class Conv3DBlock(nn.Module):
    """
    The basic block for double 3x3x3 convolutions in the analysis path
    -- __init__()
    :param in_channels -> number of input channels
    :param out_channels -> desired number of output channels
    :param bottleneck -> specifies the bottlneck block
    -- forward()
    :param input -> input Tensor to be convolved
    :return -> Tensor
    """
    def __init__(self, in_channels, out_channels, bottleneck=False) -> None:
        super(Conv3DBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=in_channels, out_channels=out_channels // 2, kernel_size=(3, 3, 3), padding=1)
        print("First conv1 layer")
        print("in_channels:", in_channels)
        print("out_channels:", out_channels // 2)
        self.bn1 = nn.BatchNorm3d(num_features=out_channels // 2)
        print("BatchNorm3d")
        print("num_features:", out_channels // 2)
        self.conv2 = nn.Conv3d(in_channels=out_channels // 2, out_channels=out_channels, kernel_size=(3, 3, 3), padding=1)
        print("Second conv2 layer")
        print("in_channels:", out_channels // 2)
        print("out_channels:", out_channels)
        self.bn2 = nn.BatchNorm3d(num_features=out_channels)
        print("BatchNorm3d")
        print("num_features:", out_channels)
        self.relu = nn.ReLU()
        self.bottleneck = bottleneck
        print("Bottleneck:", bottleneck)
        if not bottleneck:
            self.pooling = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2)

    def forward(self, input):
        res = self.relu(self.bn1(self.conv1(input)))
        #print("Shape of res:", res.shape)
        res = self.relu(self.bn2(self.conv2(res)))
        #print("Shape of res after second convolution:", res.shape)
        if not self.bottleneck:
            out = self.pooling(res)
        else:
            out = res
        return out, res



class UpConv3DBlock(nn.Module):
    """
    The basic block for upsampling followed by double 3x3x3 convolutions in the synthesis path
    -- __init__()
    :param in_channels -> number of input channels
    :param out_channels -> number of residual connections' channels to be concatenated
    :param last_layer -> specifies the last output layer
    :param num_classes -> specifies the number of output channels for dispirate classes
    -- forward()
    :param input -> input Tensor
    :param residual -> residual connection to be concatenated with input
    :return -> Tensor
    """
    def __init__(self, in_channels, res_channels=0, last_layer=False, num_classes=None) -> None:
        super(UpConv3DBlock, self).__init__()
        assert (last_layer == False and num_classes is None) or (last_layer == True and num_classes is not None), 'Invalid arguments'
        self.upconv1 = nn.ConvTranspose3d(in_channels=in_channels, out_channels=in_channels, kernel_size=(2, 2, 2), stride=2)
        print("First upconv1 layer")
        print("in_channels:", in_channels)
        print("res_channels:", res_channels)
        print("out_channels:", in_channels)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm3d(num_features=in_channels // 2)
        print("BatchNorm3d")
        print("num_features:", in_channels // 2)
        self.conv1 = nn.Conv3d(in_channels=in_channels + res_channels, out_channels=in_channels // 2, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        print("First conv1 layer")
        print("in_channels:", in_channels + res_channels)
        print("out_channels:", in_channels // 2)
        self.conv2 = nn.Conv3d(in_channels=in_channels // 2, out_channels=in_channels // 2, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        print("Second conv2 layer")
        print("in_channels:", in_channels // 2)
        print("out_channels:", in_channels // 2)
        self.last_layer = last_layer
        if last_layer:
            self.conv3 = nn.Conv3d(in_channels=in_channels // 2, out_channels=num_classes, kernel_size=(1, 1, 1))
            print("Last layer")
            print("in_channels:", in_channels)
            print("num_classes:", num_classes)

             
    def forward(self, input, residual=None):
        out = self.upconv1(input)
        #print("Shape of out after upconv1:", out.shape)
        # if residual!=None: out = torch.cat((out, residual), 1)
        if residual is not None:
            
            #print("Shape of out before concatenation:", out.shape)
            #print("Shape of residual:", residual.shape)

            # Shape of out before concatenation: torch.Size([1, 512, 8, 80, 80])
            # Shape of residual: torch.Size([1, 256, 8, 80, 80])

            # Shape of out before concatenation: torch.Size([1, 256, 16, 160, 160])
            # Shape of residual: torch.Size([1, 128, 16, 160, 160])
             
            # Shape of out before concatenation: torch.Size([1, 128, 32, 320, 320])
            # Shape of residual: torch.Size([1, 64, 32, 320, 320])

            out = torch.cat((out, residual), 1)

            #print("Shape of out after concatenation:", out.shape)

        out = self.relu(self.bn(self.conv1(out)))
        #print("Shape of out after conv1:", out.shape)
        out = self.relu(self.bn(self.conv2(out)))
        #print("Shape of out after conv2:", out.shape)
        if self.last_layer: 
            out = self.conv3(out) # Producing 'num_classes' channels
            # print("Shape of out after last layer:", out.shape)
        return out

#torchvision.models.__dict__['resnet18'](pretrained=True)

class UNet3D(nn.Module):
    """
    The 3D UNet model
    -- __init__()
    :param in_channels -> number of input channels
    :param num_classes -> specifies the number of output channels or masks for different classes
    :param level_channels -> the number of channels at each level (count top-down)
    :param bottleneck_channel -> the number of bottleneck channels 
    :param device -> the device on which to run the model
    -- forward()
    :param input -> input Tensor
    :return -> Tensor
    """
    def __init__(self, in_channels, num_classes, level_channels=[64, 128, 256], bottleneck_channel=512) -> None:
        super(UNet3D, self).__init__()
        level_1_chnls, level_2_chnls, level_3_chnls = level_channels[0], level_channels[1], level_channels[2]
        self.a_block1 = Conv3DBlock(in_channels=in_channels, out_channels=level_1_chnls)
        self.a_block2 = Conv3DBlock(in_channels=level_1_chnls, out_channels=level_2_chnls)
        self.a_block3 = Conv3DBlock(in_channels=level_2_chnls, out_channels=level_3_chnls)
        self.bottleNeck = Conv3DBlock(in_channels=level_3_chnls, out_channels=bottleneck_channel, bottleneck=True)
        self.s_block3 = UpConv3DBlock(in_channels=bottleneck_channel, res_channels=level_3_chnls)
        self.s_block2 = UpConv3DBlock(in_channels=level_3_chnls, res_channels=level_2_chnls)
        self.s_block1 = UpConv3DBlock(in_channels=level_2_chnls, res_channels=level_1_chnls, num_classes=num_classes, last_layer=True)

    def forward(self, input):
        # Analysis path forward feed
        out, residual_level1 = self.a_block1(input)
        out, residual_level2 = self.a_block2(out)
        out, residual_level3 = self.a_block3(out)
        out, _ = self.bottleNeck(out)

        # Synthesis path forward feed
        out = self.s_block3(out, residual_level3)
        out = self.s_block2(out, residual_level2)
        out = self.s_block1(out, residual_level1)
        return out

if __name__ == '__main__':
    # Configurations according to the setup
    IN_CHANNELS = cfg.IN_CHANNELS 
    NUM_CLASSES = cfg.NUM_CLASSES  

    model = UNet3D(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES)
    start_time = time.time()

    # Check model summary (optional)
    summary(model=model, input_size=(IN_CHANNELS, 16, 128, 128), batch_size=-1, device="cpu")
    print("--- %s seconds ---" % (time.time() - start_time))

"""
if __name__ == '__main__':
    #Configurations according to the Xenopus kidney dataset
    model = UNet3D(in_channels=3, num_classes=1)
    start_time = time.time()
    summary(model=model, input_size=(3, 16, 128, 128), batch_size=-1, device="cpu")
    print("--- %s seconds ---" % (time.time() - start_time))
    """
    