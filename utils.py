import torch
import torch.nn as nn
from os.path import isdir, isfile


def seed_prng(seed):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def setup_devices(no_cuda, deterministic, seed = 1):
    use_cuda = not no_cuda and torch.cuda.is_available()
    dev_count = torch.cuda.device_count() if use_cuda else 1

    if deterministic:
        seed_prng(seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    return device, dev_count


def checkpoint_state(model=None, optimizer=None, best_prec=None, epoch=None, it=None, half=False):
    optim_state = None
    if optimizer is not None:
        if isinstance(optimizer, list):
            optim_state = [opt.state_dict() for opt in optimizer]
        else:
            optim_state = optimizer.state_dict()

    if model is not None:
        if isinstance(model, torch.nn.DataParallel):
            model_state = model.module.state_dict()
        else:
            model_state = model.state_dict()
    else:
        model_state = None

    return {
        "epoch": epoch,
        "it": it,
        "best_prec": best_prec,
        "model_state": model_state,
        "optimizer_state": optim_state
    }


def save_checkpoint(state, is_best, filename="checkpoint", bestname="model_best"):
    directory = os.path.dirname(filename)
    if not os.path.isdir(directory):
        os.mkdir(directory)
    filename = "{}.pth.tar".format(filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, "{}.pth.tar".format(bestname))


def load_checkpoint(model=None, optimizer=None, filename="checkpoint"):
    filename = "{}.pth.tar".format(filename)

    if isfile(filename):
        print("==> Loading from checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        epoch = checkpoint["epoch"]
        it = checkpoint.get("it", 0.0)
        best_prec = checkpoint["best_prec"]
        if model is not None and checkpoint["model_state"] is not None:
            model.load_state_dict(checkpoint["model_state"])
        if optimizer is not None and checkpoint["optimizer_state"] is not None:
            opt_state = checkpoint["optimizer_state"]
            if isinstance(optimizer, list):
                for opt, state in zip(optimizer, opt_state):
                    opt.load_state_dict(state)
            else:
                optimizer.load_state_dict(opt_state)
        print("==> Done")
        return it, epoch, best_prec
    else:
        print("==> Checkpoint '{}' not found".format(filename))
        return None


def pixel_grid(bs, h, w, dev=None):
    px_x = torch.linspace(-1.0, 1.0, steps=w)
    px_y = torch.linspace(-1.0, 1.0, steps=h)
    px_x = px_x.unsqueeze(0).expand(h, -1).view(1, 1, h, -1).expand(bs, -1, -1, -1)
    px_y = px_y.unsqueeze(1).expand(-1, w).view(1, 1, -1, w).expand(bs, -1, -1, -1)
    if dev is not None:
        px_x = px_x.to(dev)
        px_y = px_y.to(dev)

    return px_x, px_y


class _BNBase(nn.Sequential):
    def __init__(self, in_size, batch_norm=None, name=""):
        super(_BNBase, self).__init__()
        self.add_module(name + "bn", batch_norm(in_size))

        nn.init.constant_(self[0].weight, 1.0)
        nn.init.constant_(self[0].bias, 0)


class BatchNorm2d(_BNBase):
    def __init__(self, in_size, name=""):
        # type: (BatchNorm2d, int, AnyStr) -> None
        super(BatchNorm2d, self).__init__(in_size, batch_norm=nn.BatchNorm2d, name=name)


class _ConvBase(nn.Sequential):
    def __init__(
        self,
        in_size,
        out_size,
        kernel_size,
        stride,
        padding,
        dilation,
        activation,
        bn,
        init,
        conv=None,
        norm_layer=None,
        bias=True,
        preact=False,
        gated=False,
        output_padding=0,
        groups=1,
        name="",
    ):
        super(_ConvBase, self).__init__()

        bias = bias and (not bn)
        if output_padding > 0:
            self.conv_unit = conv(
                in_size,
                out_size,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                output_padding=output_padding,
                bias=bias,
                groups=groups
            )
        else:
            self.conv_unit = conv(
                in_size,
                out_size,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=bias,
                groups=groups
            )

        init(self.conv_unit.weight)
        if bias:
            nn.init.constant_(self.conv_unit.bias, 0)

        self.preact = preact
        self.activation = activation
        self.bn_unit = None
        if bn and norm_layer is not None:
            self.bn_unit = norm_layer(out_size if not preact else in_size) if bn else None

        self.gated = gated
        if gated:
            self.mask_conv = conv(
                in_size,
                out_size,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=bias
            )
            init(self.mask_conv.weight)

    def forward(self, inp):
        if self.preact and not self.gated:
            x = inp
            if self.bn_unit is not None:
                x = self.bn_unit(x)
            if self.activation is not None:
                x = self.activation(x)

        x = self.conv_unit(inp)
        mask = None
        if self.gated:
            mask = torch.sigmoid(self.mask_conv(inp))

        if not self.preact or self.gated:
            if self.activation is not None:
                x = self.activation(x)

        if self.gated:
            x = x * mask

        if not self.preact or self.gated:
            if self.bn_unit is not None: #and inp.shape[0] > 1:
                x = self.bn_unit(x)

        return x


class Conv2d(_ConvBase):
    def __init__(
        self,
        in_size,
        out_size,
        kernel_size=(1, 1),
        stride=(1, 1),
        padding=(0, 0),
        dilation=(1, 1),
        activation=nn.ReLU(inplace=True),
        bn=False,
        init=nn.init.kaiming_normal_,
        bias=True,
        preact=False,
        gated=False,
        groups=1,
        name="",
        norm_layer=BatchNorm2d,
    ):
        # type: (Conv2d, int, int, Tuple[int, int], Tuple[int, int], Tuple[int, int], Tuple[int, int], Any, bool, Any, bool, bool, AnyStr, _BNBase) -> None
        super(Conv2d, self).__init__(
            in_size,
            out_size,
            kernel_size,
            stride,
            padding,
            dilation,
            activation,
            bn,
            init,
            conv=nn.Conv2d,
            norm_layer=norm_layer,
            bias=bias,
            preact=preact,
            gated=gated,
            name=name,
            groups=groups
        )


class ConvTranspose2d(_ConvBase):
    def __init__(
        self,
        in_size,
        out_size,
        kernel_size=(1, 1),
        stride=(1, 1),
        padding=(0, 0),
        dilation=(1, 1),
        activation=nn.ReLU(inplace=True),
        bn=False,
        init=nn.init.kaiming_normal_,
        bias=True,
        preact=False,
        output_padding=0,
        groups=1,
        name="",
        norm_layer=BatchNorm2d,
    ):
        # type: (Conv2d, int, int, Tuple[int, int], Tuple[int, int], Tuple[int, int], Tuple[int, int], Any, bool, Any, bool, bool, AnyStr, _BNBase) -> None
        super(ConvTranspose2d, self).__init__(
            in_size,
            out_size,
            kernel_size,
            stride,
            padding,
            dilation,
            activation,
            bn,
            init,
            conv=nn.ConvTranspose2d,
            norm_layer=norm_layer,
            bias=bias,
            preact=preact,
            output_padding=output_padding,
            gated=False,
            name=name,
            groups=groups
        )


class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True, activation=nn.Sigmoid()):
        super().__init__()

        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim

        padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=kernel_size,
                              padding=padding,
                              bias=bias)

        self.activation = activation

    def forward(self, input_tensor, cur_state):
        (h_cur, c_cur) = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = self.activation(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, (c_next, c_next)

    def init_hidden(self, batch_size, img_size):
        return (torch.zeros(batch_size, self.hidden_dim, *img_size).cuda(),
                torch.zeros(batch_size, self.hidden_dim, *img_size).cuda())


class ResNetLSTM(nn.Module):
    def __init__(self, in_planes, k=3, activation=None):
        super().__init__()

        self.lstm1 = ConvLSTMCell(in_planes, in_planes, kernel_size=(k,k))
        self.lstm2 = ConvLSTMCell(in_planes, in_planes, kernel_size=(k,k))
        self.activation = activation

    def forward(self, inp, state):
        state1, state2 = state
        if state1 is None:
            state1 = self.lstm1.init_hidden(inp.shape[0], inp.shape[2:])
        if state2 is None:
            state2 = self.lstm2.init_hidden(inp.shape[0], inp.shape[2:])

        out, state1 = self.lstm1(inp, state1)
        out, state2 = self.lstm2(out, state2)
        out = self.activation(out+inp) if self.activation else out+inp
        return out, (state1, state2)
