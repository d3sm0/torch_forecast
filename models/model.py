from torch import nn as nn


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.flatten(start_dim=1, end_dim=-1)


class Reshape(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)


class GlobalNormPooling(nn.Module):
    norm: int

    def __init__(self, norm=1):
        super(GlobalNormPooling, self).__init__()
        self.norm = norm

    def forward(self, x, dim=1):
        return x.norm(self.norm, dim=dim)


class GlobalAveragePooling(nn.Module):
    def __init__(self):
        super(GlobalAveragePooling, self).__init__()

    def forward(self, x, dim=1):
        return x.mean(dim=dim)


class GlobalMaxPooling(nn.Module):
    def __init__(self):
        super(GlobalMaxPooling, self).__init__()

    def forward(self, x, dim=1):
        return x.max(dim=dim)[0]


class ConvBlock(nn.Module):
    def __init__(
        self, input_size, output_size, kernel_size, stride, act=True, bias=False
    ):
        super(ConvBlock, self).__init__()
        reflection_padding, r = divmod(kernel_size - 1, 2)
        self.reflection_pad = nn.ConstantPad1d(
            (reflection_padding, reflection_padding + r), 0.0
        )
        self.conv = nn.Conv1d(input_size, output_size, kernel_size, stride, bias=bias)
        self.bn = nn.BatchNorm1d(output_size)
        self.act = nn.ELU() if act is True else nn.Identity()

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv(out)
        out = self.bn(out)
        out = self.act(out)
        return out


class ResBlock(nn.Module):
    def __init__(self, input_size, output_size):
        super(ResBlock, self).__init__()

        self.conv = nn.Sequential(
            ConvBlock(
                input_size=input_size, output_size=output_size, kernel_size=8, stride=1
            ),
            ConvBlock(
                input_size=output_size, output_size=output_size, kernel_size=5, stride=1
            ),
            ConvBlock(
                input_size=output_size,
                output_size=output_size,
                kernel_size=3,
                stride=1,
                act=False,
            ),
        )

        self.residual = ConvBlock(
            input_size=input_size,
            output_size=output_size,
            kernel_size=1,
            stride=1,
            act=False,
        )
        self.act = nn.ReLU()

    def forward(self, x):
        z = self.conv(x)
        y = self.residual(x)
        out = self.act(z + y)
        return out


class DeConvBlock(nn.Module):
    def __init__(self, input_size, output_size, kernel_size=3, p=0.2):
        super(DeConvBlock, self).__init__()
        self.conv = nn.ConvTranspose1d(input_size, output_size, kernel_size=kernel_size)
        self.up_sample = nn.Upsample()
        # self.batch_norm = nn.BatchNorm1d(num_features=output_size)
        self.act = nn.ReLU()  # PReLU(num_parameters=output_size)
        # self.dropout = nn.Dropout(p=p)

    def forward(self, x):
        x = self.conv(x)

        # x = self.max_pool(x)
        # x = self.batch_norm(x)
        x = self.act(x)
        # x = self.dropout(x)
        return x


class LinearBlock(nn.Module):
    def __init__(self, input_size, output_size, p=0.2):
        super(LinearBlock, self).__init__()
        self.fc = nn.Linear(input_size, output_size)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(p)

    def forward(self, x):
        x = self.fc(x)
        x = self.act(x)
        x = self.dropout(x)
        return x


class ResNet(nn.Module):
    def __init__(self, input_size=2, seq_len=1, hidden_dim=1, output_size=1):
        super(ResNet, self).__init__()

        self.b1 = nn.Sequential(
            ResBlock(input_size, output_size=hidden_dim),
            # ResBlock(hidden_dim * 2, output_size=hidden_dim * 2),
            # ResBlock(hidden_dim * 2, output_size=hidden_dim * 2),
            GlobalMaxPooling(),
        )
        self.out = nn.Linear(seq_len, output_size)

    def forward(self, x):
        x = self.conv(x)
        x = self.out(x)
        return x


class ConvNet(nn.Module):
    def __init__(self, input_size=2, seq_len=1, output_size=1, hidden_size=8):
        super(ConvNet, self).__init__()

        self.conv = nn.Sequential(
            ConvBlock(
                input_size=input_size, output_size=hidden_size, kernel_size=8, stride=1
            ),
            ConvBlock(
                input_size=hidden_size,
                output_size=hidden_size * 2,
                kernel_size=5,
                stride=1,
            ),
            ConvBlock(
                input_size=hidden_size * 2,
                output_size=hidden_size,
                kernel_size=3,
                stride=1,
            ),
            GlobalMaxPooling(),
        )

        self.out = nn.Linear(seq_len, output_size)

    def forward(self, x):
        x = self.conv(x)
        x = self.out(x)
        return x


class DeConvNet(nn.Module):
    def __init__(self, input_channel=8, num_classes=7):
        super(DeConvNet, self).__init__()

        self.network = nn.Sequential(
            *[
                LinearBlock(input_size=num_classes, output_size=420),
                Reshape(-1, 10, 42),
                DeConvBlock(input_size=10, output_size=32, kernel_size=5),
                DeConvBlock(input_size=32, output_size=input_channel, kernel_size=5),
            ]
        )

    def forward(self, x):
        x = self.network(x)
        return x


class LinearNet(nn.Module):
    def __init__(self, input_channel=3, output_channel=1, hidden_size=32):
        super(LinearNet, self).__init__()

        self.project = nn.Sequential(Flatten(), nn.Dropout(p=0.1))
        self.fc = nn.Sequential(
            LinearBlock(input_size=input_channel, output_size=hidden_size, p=0.2),
            LinearBlock(input_size=hidden_size, output_size=hidden_size, p=0.2),
            LinearBlock(input_size=hidden_size, output_size=hidden_size, p=0.3),
        )

        self.out = nn.Linear(hidden_size, output_channel)

    def forward(self, x):
        h = self.project(x)
        h = self.fc(h)
        y = self.out(h)
        return y


class Baseline(nn.Module):
    def __init__(self):
        super(Baseline, self).__init__()

    def forward(self, x):
        x = x[:, :, 1:].sum(dim=-1, keepdim=True)
        return x
