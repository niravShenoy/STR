import torch.nn as nn
import torch

from utils.builder import get_builder
from args import args

# BasicBlock {{{
class BasicBlock(nn.Module):
    M = 2
    expansion = 1

    def __init__(self, builder, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = builder.conv3x3(inplanes, planes, stride)
        self.bn1 = builder.batchnorm(planes)
        self.relu = builder.activation()
        self.conv2 = builder.conv3x3(planes, planes)
        self.bn2 = builder.batchnorm(planes, last_bn=True)
        downsample = None
        if stride != 1 or inplanes != planes * self.expansion:
            dconv = builder.conv1x1(
                inplanes, planes * self.expansion, stride=stride
            )
            dbn = builder.batchnorm(planes * self.expansion)
            if dbn is not None:
                downsample = nn.Sequential(dconv, dbn)
            else:
                downsample = dconv
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        if self.bn1 is not None:
            out = self.bn1(out)

        out = self.relu(out)

        out = self.conv2(out)

        if self.bn2 is not None:
            out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


# BasicBlock }}}

# Bottleneck {{{
class Bottleneck(nn.Module):
    M = 3
    expansion = 4

    def __init__(self, builder, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = builder.conv1x1(inplanes, planes)
        self.bn1 = builder.batchnorm(planes)
        self.conv2 = builder.conv3x3(planes, planes, stride=stride)
        self.bn2 = builder.batchnorm(planes)
        self.conv3 = builder.conv1x1(planes, planes * self.expansion)
        self.bn3 = builder.batchnorm(planes * self.expansion, last_bn=True)
        self.relu = builder.activation()
        downsample = None
        if stride != 1 or inplanes != planes * self.expansion:
            dconv = builder.conv1x1(
                inplanes, planes * self.expansion, stride=stride
            )
            dbn = builder.batchnorm(planes * self.expansion)
            if dbn is not None:
                downsample = nn.Sequential(dconv, dbn)
            else:
                downsample = dconv
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        out = self.relu(out)

        return out


# Bottleneck }}}

# ResNet {{{
class ResNet(nn.Module):
    def __init__(self, builder, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        if args.first_layer_dense:
            print("FIRST LAYER DENSE!!!!")
            self.conv1 = nn.Conv2d(
                3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False
            )
        else:
            self.conv1 = builder.conv7x7(3, 64, stride=2, first_layer=True)

        self.bn1 = builder.batchnorm(64)
        self.relu = builder.activation()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(builder, block, 64, layers[0])
        self.layer2 = self._make_layer(builder, block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(builder, block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(builder, block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        # self.fc = nn.Linear(512 * block.expansion, num_classes)
        if args.last_layer_dense:
            self.fc = nn.Conv2d(512 * block.expansion, args.num_classes, 1)
        else:
            self.fc = builder.conv1x1(512 * block.expansion, num_classes)

    def _make_layer(self, builder, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            dconv = builder.conv1x1(
                self.inplanes, planes * block.expansion, stride=stride
            )
            dbn = builder.batchnorm(planes * block.expansion)
            if dbn is not None:
                downsample = nn.Sequential(dconv, dbn)
            else:
                downsample = dconv

        layers = []
        layers.append(block(builder, self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(builder, self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        torch.cuda.empty_cache()
        x = self.conv1(x)
        if self.bn1 is not None:
            x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = self.fc(x)
        x = x.view(x.size(0), -1)

        return x


# ResNet }}}
def ResNet18(num_classes, pretrained=True):
    # TODO: pretrained
    return ResNet(get_builder(), BasicBlock, [2, 2, 2, 2], num_classes)


def ResNet50(num_classes, pretrained=True):
    # TODO: pretrained
    return ResNet(get_builder(), Bottleneck, [3, 4, 6, 3], num_classes)

class ResNetWidth(nn.Module):
    """
    ResNetWidth is a variant of the ResNet model with adjustable width.

    Args:
        builder (object): The builder object used to construct the model.
        block (object): The block object representing the basic building block of the model.
        num_blocks (list): A list of integers representing the number of blocks in each layer of the model.
        width (int): The width of the model, which determines the number of channels in the convolutional layers.
        num_classes (int, optional): The number of output classes. Defaults to 10.
    """

    def __init__(self, builder, block, num_blocks, width, num_classes=10):
        super(ResNetWidth, self).__init__()
        self.in_planes = width

        self.conv1 = builder.conv3x3(3, width)
        
        self.bn1 = builder.batchnorm(width)
        self.relu = builder.activation()
        self.layer1 = self._make_layer(builder, block, width, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(builder, block, width, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(builder, block, width, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(builder, block, width, num_blocks[3], stride=2)
        self.linear = builder.conv1x1(width*block.expansion, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def _make_layer(self, builder, block, planes, num_blocks, stride):
        """
        Helper method to create a layer of blocks.

        Args:
            builder (object): The builder object used to construct the model.
            block (object): The block object representing the basic building block of the model.
            planes (int): The number of output channels for each block in the layer.
            num_blocks (int): The number of blocks in the layer.
            stride (int): The stride value for the layer.

        Returns:
            nn.Sequential: A sequential container of blocks representing the layer.
        """
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(builder, self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        torch.cuda.empty_cache()
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = self.linear(out).squeeze()

        return out


def ResNetWidth18(input_shape, num_classes, width=64, dense_classifier=False, pretrained=True):
    return ResNetWidth(get_builder(), BasicBlock, [2, 2, 2, 2], width, num_classes)


def ResNetWidth34(input_shape, num_classes, width=128, dense_classifier=False, pretrained=True):
    return ResNetWidth(get_builder(), BasicBlock, [3, 4, 6, 3], width, num_classes)


def ResNetWidth50(input_shape, num_classes, width=128, dense_classifier=False, pretrained=True):
    return ResNetWidth(get_builder(), Bottleneck, [3, 4, 6, 3], width, num_classes)