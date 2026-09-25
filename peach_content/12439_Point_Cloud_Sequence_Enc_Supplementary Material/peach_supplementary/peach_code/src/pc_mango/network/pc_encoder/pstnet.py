import torch
import torch.nn as nn
import torch.nn.functional as F

from hydra import initialize, compose
from torchinfo import summary


class PSTNet(nn.Module):
    def __init__(self, config, example_batch):
        radius = config.radius
        nsamples = config.nsamples
        output_dim = config.output_dim
        self.config = config

        super().__init__()
        self.dummy_mode = False
        try:
            from pc_mango.network.util.pstnet.modules.pst_convolutions import PSTConv
            print("PSTConv imported successfully.")
        except ImportError:
            print("Failed to import PSTConv. Ensure that pc_mango is installed correctly. Dummy Mode activated.")
            print("WARNING: The network will not function properly in Dummy Mode.")
            self.dummy_mode = True

        if not self.dummy_mode:
            # TODO: Correctly process color information, right now it is just ignored
            self.conv1 =  PSTConv(in_planes=0,
                                  mid_planes=16,
                                  out_planes=32,
                                  spatial_kernel_size=[radius, nsamples],
                                  temporal_kernel_size=1,
                                  spatial_stride=2,
                                  temporal_stride=1,
                                  temporal_padding=[0,0],
                                  spatial_aggregation="multiplication",
                                  spatial_pooling="sum")

            self.conv2 = PSTConv(in_planes=32,
                                 mid_planes=48,
                                 out_planes=64,
                                 spatial_kernel_size=[2*radius, nsamples],
                                 temporal_kernel_size=3,
                                 spatial_stride=2,
                                 temporal_stride=2,
                                 temporal_padding=[1,0],
                                 spatial_aggregation="multiplication",
                                 spatial_pooling="sum")

            self.conv3 = PSTConv(in_planes=64,
                                 mid_planes=96,
                                 out_planes=128,
                                 spatial_kernel_size=[2*2*radius, nsamples],
                                 temporal_kernel_size=3,
                                 spatial_stride=2,
                                 temporal_stride=1,
                                 temporal_padding=[1,0],
                                 spatial_aggregation="multiplication",
                                 spatial_pooling="sum")


            self.conv4 =  PSTConv(in_planes=128,
                                  mid_planes=192,
                                  out_planes=256,
                                  spatial_kernel_size=[2*2*radius, nsamples],
                                  temporal_kernel_size=1,
                                  spatial_stride=2,
                                  temporal_stride=1,
                                  temporal_padding=[0,0],
                                  spatial_aggregation="multiplication",
                                  spatial_pooling="sum")

            self.fc = nn.Linear(256, output_dim)

    def forward(self, xyzs, colors):
        if self.dummy_mode:
            return torch.zeros(xyzs.shape[0], self.config.output_dim).to(xyzs.device)
        if xyzs.shape[1] == 25 or xyzs.shape[1] == 51:
            # add one dummy dimension so that network works with 26/52 timesteps as expected
            xyzs = torch.cat([xyzs, torch.zeros_like(xyzs[:, :1, :, :])], dim=1)
            colors = torch.cat([colors, torch.zeros_like(colors[:, :1, :, :])], dim=1)
        new_xys, new_features = self.conv1(xyzs, colors)
        new_features = F.relu(new_features)

        new_xys, new_features = self.conv2(new_xys, new_features)
        new_features = F.relu(new_features)


        new_xys, new_features = self.conv3(new_xys, new_features)
        new_features = F.relu(new_features)


        new_xys, new_features = self.conv4(new_xys, new_features)               # (B, L, C, N)

        new_features = torch.mean(input=new_features, dim=-1, keepdim=False)    # (B, L, C)

        new_feature = torch.max(input=new_features, dim=1, keepdim=False)[0]    # (B, C)

        out = self.fc(new_feature)

        return out, None, None # two outputs needed for same interface as pointpatch encoder


if __name__ == "__main__":
    with initialize(config_path="../../../../config/network/pc_encoder", version_base="1.3.2"):
        config = compose(config_name="pstnet")

    xyzs = torch.zeros(1, 26, 512, 3).cuda()
    colors = torch.ones(1, 26, 512, 1).cuda()
    model = PSTNet(config, None).to("cuda")
    # print(summary(model, xyzs.shape))
    out = model(xyzs, colors)
    print(out.shape)