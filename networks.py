from typing import List
import torch
import torch.nn as nn
from torch import Tensor


def argmax2d(x: Tensor) -> Tensor:
    '''Given tensor with shape (B, C, H, W) returns a long tensor of shape (B,C,2)
    with the indices of the maximum value in each feature map of the batch
    '''
    flat_idxs = torch.max( torch.flatten(x, 2), dim=2 )[1]
    h_idxs = torch.div(flat_idxs, x.size(2)) # equiv to //
    w_idxs = flat_idxs % x.size(2)

    return torch.stack([h_idxs, w_idxs], dim=-1)

def argmax3d(x: Tensor) -> Tensor:
    '''Given tensor with shape (B, C, H, W) returns a long tensor of shape (B,3)
    with the indices of the maximum value in each feature map of the batch
    '''
    flat_idxs = torch.max( torch.flatten(x, 1), dim=1 )[1]
    w_idxs = flat_idxs % x.size(3)
    flat_idxs = torch.div(flat_idxs, x.size(3))
    h_idxs = flat_idxs % x.size(2)  # equiv to //
    flat_idxs = torch.div(flat_idxs, x.size(2))
    c_idxs = flat_idxs % x.size(1)

    return torch.stack([c_idxs, h_idxs, w_idxs], dim=-1)



class PixelWiseQNetwork(nn.Module):
    def __init__(self, img_shape: List[int]):
        '''Q-Network that predicts action values corresponding to each pixel
        in the input image

        Parameters
        ----------
        img_shape
            shape of input image (C, H, W)
        '''
        super().__init__()

        self.conv0 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=2)
        self.conv1 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2)

        self.conv3 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=2)
        self.conv4 = nn.ConvTranspose2d(in_channels=32, out_channels=8, kernel_size=3, stride=2, output_padding=1)
        self.conv5 = nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=3, stride=2, output_padding=1)
        self.conv6 = nn.Conv2d(in_channels=8, out_channels=6, kernel_size=3, padding=1)

        self.relu = nn.ReLU(inplace=True)
        self.loss_fn = nn.MSELoss()


    def forward(self, x: Tensor) -> Tensor:
        '''
        Parameters
        ----------
        x
            float image tensor of shape (B, C, H, W)

        Returns
        -------
        q_map
            float tensor of shape (B, H, W ), for each pixel in input image
            the q-network predicts the value of perfoming grasp at said pixel
            location
        '''

        x_down1 = self.relu(self.conv0(x))
        x_down2 = self.relu(self.conv1(x_down1))
        x_down3 = self.relu(self.conv2(x_down2))

        x_up1 = self.relu(self.conv3(x_down3))
        x_up2 = self.relu(self.conv4(torch.cat([x_down2, x_up1], dim=1)))
        x_up3 = self.relu(self.conv5(torch.cat([x_down1, x_up2], dim=1)))
        
        x_out = self.conv6(x_up3)

        return x_out


    @torch.no_grad()
    def predict(self, x: Tensor) -> Tensor:
        '''Predicts action (px,py) for each state by taking 2dargmax of qmap
        predictions.  Gradient calculation is disabled

        Parameters
        ----------
        x
            float image tensor of shape (B, C, H, W)

        Returns
        -------
        action
            integer tensor of shape (B, 2) indicating action with maximum value
            along height and width dimensions of image
        '''
        q_map = self.forward(x)
        return argmax3d(q_map)

    def compute_loss(self, q_pred: Tensor, q_target: Tensor) -> Tensor:
        return self.loss_fn(q_pred, q_target)


if __name__ == "__main__":
    inputs = torch.zeros((1, 3, 42, 42), dtype=torch.float32)

    net = PixelWiseQNetwork((3, 42, 42))

    outputs = net.forward(inputs)

    assert outputs.shape[1] == 1, \
        'Output qmap should have 1 channel dimension'

    assert inputs.shape[2:] == outputs.shape[2:], \
        'Input and output image dimensions should be the same'

    print('It appears your model is working properly. \n'
          'Double check that you used ReLUs between layers.')

