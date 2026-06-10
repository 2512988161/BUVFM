import torch
from torch.nn import functional as F
from torchvision import transforms, datasets
from torchdiffeq import odeint


class Config:
    epoch = 200
    batchsize = 50
    num_workers = 3
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


config = Config()

torch.manual_seed(42)


train_transforms = transforms.Compose(
    [
        # transforms.Pad(2),
        # transforms.RandomCrop(28),
        # transforms.RandomAffine(degrees=15),
        transforms.ToTensor(),
    ]
)
test_transforms = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST("mnist", train=True, transform=train_transforms, download=True),
    batch_size=config.batchsize,
    shuffle=True,
    num_workers=config.num_workers,
)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST("mnist", train=False, transform=test_transforms, download=True),
    batch_size=config.batchsize,
    shuffle=False,
    num_workers=config.num_workers,
)


class neuronODE(torch.nn.Module):
    def __init__(self):
        super(neuronODE, self).__init__()
        self.gamma = None

    def fresh(self, gamma):
        self.gamma = gamma

    def forward(self, t, p):
        dpdt = -p + torch.pow(torch.sin(p + self.gamma), 2)
        return dpdt


class forwardinvODE(torch.nn.Module):
    def __init__(self):
        super(forwardinvODE, self).__init__()
        self.gamma = None

    def fresh(self, gamma):
        self.gamma = gamma

    def forward(self, t, all):
        ps, lambdas, etas = all
        dpds = ps - torch.pow(torch.sin(ps + self.gamma), 2)
        dlambdads = -lambdas * (1 - torch.sin(2 * (ps + self.gamma)))
        detads = lambdas * torch.sin(2 * (ps + self.gamma))
        return (dpds, dlambdads, detads)


class nmODE:
    def __init__(self, xsize, ysize, asize, alpha, beta, tbar, t0, step):
        self.neuronODE = neuronODE()
        self.forwardinvODE = forwardinvODE()
        self.xsize = xsize
        self.ysize = ysize
        self.asize = asize
        self.w1 = torch.zeros(ysize, xsize, device=config.device)
        self.w2 = torch.zeros(asize, ysize, device=config.device)
        self.b = torch.zeros(ysize, device=config.device)
        self.alpha = alpha
        self.beta = beta
        self.tbar = tbar
        self.t0 = t0
        self.step = step
        torch.nn.init.uniform_(self.w1, -0.1, 0.1)
        torch.nn.init.uniform_(self.w2, -0.1, 0.1)

    @torch.no_grad()
    def train(self, batch):
        # init
        cnt = 0.0
        batch[0] = batch[0].to(config.device)
        batch[1] = batch[1].to(config.device)
        batch_size = batch[1].shape[0]
        t = torch.tensor([self.t0, self.tbar], device=config.device)
        y0 = torch.zeros(batch_size, self.ysize, device=config.device)  # b, ysize
        yt = torch.zeros(batch_size, self.ysize, device=config.device)  # b, ysize
        etat = torch.zeros(batch_size, self.ysize, device=config.device)  # b, ysize
        eta0 = torch.zeros(batch_size, self.ysize, device=config.device)  # b, ysize

        x = batch[0].view(batch_size, -1)  # b, xsize
        d = batch[1]  # b, 1

        # forward
        # (b, xsize) x (xsize, ysize) + (ysize,) -> (b, ysize)
        gamma = torch.matmul(x, self.w1.T) + self.b
        for i in range(0, self.ysize, self.step):
            inext = min(i + self.step, self.ysize)
            self.neuronODE.fresh(gamma[:, i:inext])  # b, step
            yt[:, i:inext] = odeint(self.neuronODE, y0[:, i:inext], t)[-1]  # b, step

        zt = torch.matmul(yt, self.w2.T)  # (b, ysize) x (ysize, asize) -> (b, asize)
        zt_ = zt - zt.max(dim=1, keepdim=True)[0]  # b, asize
        at = F.softmax(zt_, dim=1)  # b, asize

        cnt += (at.argmax(dim=1) == d).sum()

        # backward
        deltazt = at - F.one_hot(d, num_classes=self.asize)  # b, asize
        deltazt /= batch_size

        # (b, asize) x (asize, ysize) -> (b, ysize)
        lambdat = torch.matmul(deltazt, self.w2)
        for i in range(0, self.ysize, self.step):
            inext = min(i + self.step, self.ysize)
            self.forwardinvODE.fresh(gamma[:, i:inext])  # b, step
            _, _, etaall = odeint(
                self.forwardinvODE,
                (yt[:, i:inext], lambdat[:, i:inext], etat[:, i:inext]),
                t,
            )  # t, b, step
            eta0[:, i:inext] = etaall[-1]  # b, step

        # update
        # (asize, b) x (b, ysize) -> (asize, ysize)
        self.w2 -= self.beta * torch.matmul(deltazt.T, yt)
        # (ysize, b) x (b, xsize) -> (ysize, xsize)
        self.w1 -= self.alpha * torch.matmul(eta0.T, x)
        # (b, ysize) -> (ysize,)
        self.b -= self.alpha * eta0.sum(dim=0)

        batch_acc = cnt / batch_size
        return batch_acc

    @torch.no_grad()
    def test(self, batch):
        # init
        batch[0] = batch[0].to(config.device)
        batch_size = batch[1].shape[0]
        t = torch.tensor([self.t0, self.tbar], device=config.device)
        y0 = torch.zeros(batch_size, self.ysize, device=config.device)  # b, ysize
        yt = torch.zeros(batch_size, self.ysize, device=config.device)  # b, ysize

        x = batch[0].view(batch_size, -1)  # b, xsize

        # forward
        # (b, xsize) x (xsize, ysize) + (ysize,) -> (b, ysize)
        gamma = torch.matmul(x, self.w1.T) + self.b
        for i in range(0, self.ysize, self.step):
            inext = min(i + self.step, self.ysize)
            self.neuronODE.fresh(gamma[:, i:inext])  # b, step
            yt[:, i:inext] = odeint(self.neuronODE, y0[:, i:inext], t)[-1]  # b, step

        zt = torch.matmul(yt, self.w2.T)  # (b, ysize) x (ysize, asize) -> (b, asize)
        zt_ = zt - zt.max(dim=1, keepdim=True)[0]  # b, asize
        at = F.softmax(zt_, dim=1)  # b, asize
        return at.detach()

    def save(self, path):
        torch.save({"w1": self.w1, "b": self.b, "w2": self.w2}, path)

    def load(self, path):
        params = torch.load(path, map_location=config.device)
        self.w1 = params["w1"]
        self.w2 = params["w2"]
        self.b = params["b"]


if __name__ == "__main__":
    net = nmODE(
        xsize=784, ysize=2048, asize=10, alpha=0.1, beta=0.1, tbar=5.0, t0=0, step=2048
    )
    for epoch_id in range(config.epoch):
        if epoch_id % 100 == 0:
            print(epoch_id)
        for batch_id, batch in enumerate(train_loader):
            batch_acc = net.train(batch)
            print(f"Epoch: {epoch_id} Batch: {batch_id} batch_acc={batch_acc:.4f}")

        net.save(f"{epoch_id}_params.pth")

        total, correct = 0.0, 0.0
        for batch in test_loader:
            a_pred = net.test(batch).cpu().argmax(dim=1)
            a_true = batch[1]
            correct += (a_pred == a_true).sum().item()
            total += a_true.shape[0]
        print(f"Epoch: {epoch_id} Test acc={correct/total:.4f}")
