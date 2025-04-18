import torch
from torch import nn

class TinyVGG(nn.Module):
  """Creates the TinyVGG architecture.

  Replicates the TinyVGG architecture from the CNN explainer website in PyTorch.
  See the original architecture here: https://poloclub.github.io/cnn-explainer/

  Args:
    input_shape: An integer indicating number of input channels.
    hidden_units: An integer indicating number of hidden units between layers.
    output_shape: An integer indicating number of output units.
  """
  def __init__(self, input_shape: int, hidden_units: int, output_shape: int) -> None:
      super().__init__()
      self.conv_block_1 = nn.Sequential(
          nn.Conv2d(in_channels=input_shape, 
                    out_channels=hidden_units, 
                    kernel_size=3, 
                    stride=1, 
                    padding=0),  
          nn.ReLU(),
          nn.Conv2d(in_channels=hidden_units, 
                    out_channels=hidden_units,
                    kernel_size=3,
                    stride=1,
                    padding=0),
          nn.ReLU(),
          nn.MaxPool2d(kernel_size=2,
                        stride=2)
      )
      self.conv_block_2 = nn.Sequential(
          nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=0),
          nn.ReLU(),
          nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=0),
          nn.ReLU(),
          nn.MaxPool2d(2)
      )
      self.classifier = nn.Sequential(
          nn.Flatten(),
          # Where did this in_features shape come from? 
          # It's because each layer of our network compresses and changes the shape of our inputs data.
          nn.Linear(in_features=hidden_units*13*13,
                    out_features=output_shape)
      )

  def forward(self, x: torch.Tensor):
      x = self.conv_block_1(x)
      x = self.conv_block_2(x)
      x = self.classifier(x)
      return x
      return self.classifier(self.conv_block_2(self.conv_block_1(x))) # <- leverage the benefits of operator fusion



# class BasicBlock(nn.Module):
#     def __init__(
#         self, 
#         in_channels: int,
#         out_channels: int,
#         stride: int = 1,
#         expansion: int = 1,
#         downsample: nn.Module = None
#     ) -> None:
#         super(BasicBlock, self).__init__()
#         # Multiplicative factor for the subsequent conv2d layer's output channels.
#         # It is 1 for ResNet18 and ResNet34.
#         self.expansion = expansion
#         self.downsample = downsample
#         self.conv1 = nn.Conv2d(
#             in_channels, 
#             out_channels, 
#             kernel_size=3, 
#             stride=stride, 
#             padding=1,
#             bias=False
#         )
#         self.bn1 = nn.BatchNorm2d(out_channels)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = nn.Conv2d(
#             out_channels, 
#             out_channels*self.expansion, 
#             kernel_size=3, 
#             padding=1,
#             bias=False
#         )
#         self.bn2 = nn.BatchNorm2d(out_channels*self.expansion)
#     def forward(self, x: Tensor) -> Tensor:
#         identity = x
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#         out = self.conv2(out)
#         out = self.bn2(out)
#         if self.downsample is not None:
#             identity = self.downsample(x)
#         out += identity
#         out = self.relu(out)
#         return  out

# class ResNet(nn.Module):
#     def __init__(
#         self, 
#         img_channels: int,
#         num_layers: int,
#         block: Type[BasicBlock],
#         num_classes: int  = 1000
#     ) -> None:
#         super(ResNet, self).__init__()
#         if num_layers == 18:
#             # The following `layers` list defines the number of `BasicBlock` 
#             # to use to build the network and how many basic blocks to stack
#             # together.
#             layers = [2, 2, 2, 2]
#             self.expansion = 1
        
#         self.in_channels = 64
#         # All ResNets (18 to 152) contain a Conv2d => BN => ReLU for the first
#         # three layers. Here, kernel size is 7.
#         self.conv1 = nn.Conv2d(
#             in_channels=img_channels,
#             out_channels=self.in_channels,
#             kernel_size=7, 
#             stride=2,
#             padding=3,
#             bias=False
#         )
#         self.bn1 = nn.BatchNorm2d(self.in_channels)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         self.layer1 = self._make_layer(block, 64, layers[0])
#         self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
#         self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
#         self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc = nn.Linear(512*self.expansion, num_classes)
#     def _make_layer(
#         self, 
#         block: Type[BasicBlock],
#         out_channels: int,
#         blocks: int,
#         stride: int = 1
#     ) -> nn.Sequential:
#         downsample = None
#         if stride != 1:
#             """
#             This should pass from `layer2` to `layer4` or 
#             when building ResNets50 and above. Section 3.3 of the paper
#             Deep Residual Learning for Image Recognition
#             (https://arxiv.org/pdf/1512.03385v1.pdf).
#             """
#             downsample = nn.Sequential(
#                 nn.Conv2d(
#                     self.in_channels, 
#                     out_channels*self.expansion,
#                     kernel_size=1,
#                     stride=stride,
#                     bias=False 
#                 ),
#                 nn.BatchNorm2d(out_channels * self.expansion),
#             )
#         layers = []
#         layers.append(
#             block(
#                 self.in_channels, out_channels, stride, self.expansion, downsample
#             )
#         )
#         self.in_channels = out_channels * self.expansion
#         for i in range(1, blocks):
#             layers.append(block(
#                 self.in_channels,
#                 out_channels,
#                 expansion=self.expansion
#             ))
#         return nn.Sequential(*layers)
#     def forward(self, x: Tensor) -> Tensor:
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
#         # The spatial dimension of the final layer's feature 
#         # map should be (7, 7) for all ResNets.
#         print('Dimensions of the last convolutional feature map: ', x.shape)
#         x = self.avgpool(x)
#         x = torch.flatten(x, 1)
#         x = self.fc(x)
#         return x
