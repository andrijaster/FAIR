import torch, os
import torch.utils.data
import numpy as np
from torch import nn
from torch.nn import functional as F


class CLFR_class:
    def __init__(
        self,
        input_size,  # input size
        num_layers_z,  # no. layers in first network
        num_layers_y,  # no. layers in sensitive and output networks
        step_z,  # step in first network
        step_y,  # step in sensitive and output networks
        name="FAD",  # name of model
        save_dir=None,  # directory where weights should be saved
    ):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLFR_class.FairClass(
            input_size, num_layers_z, num_layers_y, step_z, step_y
        )
        self.model.to(self.device)
        self.path = os.path.join(save_dir, name)

    class FairClass(nn.Module):
        def __init__(self, inp_size, num_layers_z, num_layers_y, step_z, step_y):
            super(CLFR_class.FairClass, self).__init__()

            num_layers_A = num_layers_y
            lst_z = nn.ModuleList()
            lst_1 = nn.ModuleList()
            lst_2 = nn.ModuleList()
            lst_3 = nn.ModuleList()
            out_size = inp_size

            for i in range(num_layers_z):
                inp_size = out_size
                out_size = int(inp_size // step_z)
                block = nn.Sequential(
                    nn.Linear(inp_size, out_size),
                    nn.BatchNorm1d(num_features=out_size),
                    nn.ReLU(),
                )
                lst_z.append(block)

            out_size_old = out_size
            for i in range(num_layers_y):
                inp_size = out_size
                out_size = int(inp_size // step_y)
                if i == num_layers_y - 1:
                    block = nn.Linear(inp_size, 1)
                else:
                    block = nn.Sequential(
                        nn.Linear(inp_size, out_size),
                        nn.BatchNorm1d(num_features=out_size),
                        nn.ReLU(),
                    )
                lst_1.append(block)

            out_size = out_size_old
            for i in range(num_layers_A):
                inp_size = out_size
                out_size = int(inp_size // step_y)
                if i == num_layers_y - 1:
                    block = nn.Linear(inp_size, 1)
                else:
                    block = nn.Sequential(
                        nn.Linear(inp_size, out_size),
                        nn.BatchNorm1d(num_features=out_size),
                        nn.ReLU(),
                    )
                lst_2.append(block)

            out_size = out_size_old
            for i in range(num_layers_A):
                inp_size = out_size
                out_size = int(inp_size // step_y)
                if i == num_layers_y - 1:
                    block = nn.Linear(inp_size, 1)
                else:
                    block = nn.Sequential(
                        nn.Linear(inp_size, out_size),
                        nn.BatchNorm1d(num_features=out_size),
                        nn.ReLU(),
                    )
                lst_3.append(block)

            self.fc1 = nn.Sequential(*lst_z)
            self.fc2 = nn.Sequential(*lst_1)
            self.fc3 = nn.Sequential(*lst_2)
            self.fc4 = nn.Sequential(*lst_3)

        def forward(self, x):
            z = self.fc1(x)
            y = torch.sigmoid(self.fc2(z))
            A_1 = torch.sigmoid(self.fc3(z))
            A_0 = torch.sigmoid(self.fc3(z))
            return y, A_1, A_0

    def fit(
        self,
        dataloader,  # train dataloader
        dataloader_val,  # validation dataloader
        early_stopping_no=3,  # early stopping no.
        max_epoch=300,  # max_epochs
        alpha=1,  # hyperparameter \alpha
        log=1,  # evaluate validation loss {1 - Yes, 0 - No}
        log_epoch=1,  # no. of epoch for evaluation of validation loss
        learning_rate=0.0001,  # learning rate
    ):

        nll_criterion = F.binary_cross_entropy
        list_1 = list(self.model.fc1.parameters()) + list(self.model.fc2.parameters())
        list_2 = list(self.model.fc3.parameters()) + list(self.model.fc4.parameters())
        optimizer_1 = torch.optim.Adam(list_1, lr=learning_rate)
        optimizer_2 = torch.optim.Adam(list_2, lr=learning_rate)

        prev_loss_y = 9e10
        no_val = 0

        for e in range(max_epoch):
            self.model.train()
            for batch_x, batch_y, batch_A in dataloader:

                batch_x = batch_x.to(self.device, dtype=torch.float)
                batch_y = batch_y.unsqueeze(dim=1).to(self.device, dtype=torch.float)
                batch_A = batch_A.unsqueeze(dim=1).to(self.device, dtype=torch.float)
                batch_A_1 = batch_A[batch_y == 1]
                batch_A_0 = batch_A[batch_y == 0]
                y_pred, A_1_pred, A_0_pred = self.model(batch_x)
                A_1_pred = A_1_pred[batch_y == 1]
                A_0_pred = A_0_pred[batch_y == 0]

                loss2 = A_1_pred.shape[0] / batch_A.shape[0] * nll_criterion(
                    A_1_pred, batch_A_1
                ) + A_0_pred.shape[0] / batch_A.shape[0] * nll_criterion(
                    A_0_pred, batch_A_0
                )
                optimizer_2.zero_grad()
                loss2.backward()
                optimizer_2.step()

                y_pred, A_1_pred, A_0_pred = self.model(batch_x)
                A_1_pred = A_1_pred[batch_y == 1]
                A_0_pred = A_0_pred[batch_y == 0]

                loss1 = nll_criterion(y_pred, batch_y) - alpha * (
                    A_1_pred.shape[0]
                    / batch_A.shape[0]
                    * nll_criterion(A_1_pred, batch_A_1)
                    + A_0_pred.shape[0]
                    / batch_A.shape[0]
                    * nll_criterion(A_0_pred, batch_A_0)
                )
                optimizer_1.zero_grad()
                loss1.backward()
                optimizer_1.step()

            if e % log_epoch == 0 and log == 1:
                self.model.eval()
                for batch_x, batch_y, batch_A in dataloader_val:

                    batch_x = batch_x.to(self.device, dtype=torch.float)
                    batch_y = batch_y.unsqueeze(dim=1).to(
                        self.device, dtype=torch.float
                    )
                    batch_A = batch_A.unsqueeze(dim=1).to(
                        self.device, dtype=torch.float
                    )
                    batch_A_1 = batch_A[batch_y == 1]
                    batch_A_0 = batch_A[batch_y == 0]
                    y_pred, A_1_pred, A_0_pred = self.model(batch_x)
                    A_1_pred = A_1_pred[batch_y == 1]
                    A_0_pred = A_0_pred[batch_y == 0]

                    loss2 = A_1_pred.shape[0] / batch_A.shape[0] * nll_criterion(
                        A_1_pred, batch_A_1
                    ) + A_0_pred.shape[0] / batch_A.shape[0] * nll_criterion(
                        A_0_pred, batch_A_0
                    )
                    loss2 = loss2.data.cpu().numpy()

                    loss1 = nll_criterion(y_pred, batch_y) - alpha * (
                        A_1_pred.shape[0]
                        / batch_A.shape[0]
                        * nll_criterion(A_1_pred, batch_A_1)
                        + A_0_pred.shape[0]
                        / batch_A.shape[0]
                        * nll_criterion(A_0_pred, batch_A_0)
                    )
                    loss1 = loss1.data.cpu().numpy()

                    if loss1 > prev_loss_y:
                        no_val += 1
                    else:
                        prev_loss_y, _ = loss1, loss2
                        torch.save(self.model.state_dict(), self.path)
                        print("Model saved")
                        no_val = 0

                if no_val == early_stopping_no:
                    break

    def predict(self, x_test):  # Inference
        self.model.eval()
        y, A = self.model(x_test)
        y = np.round(y.data)
        A = np.round(A.data)
        return y, A

    def predict_proba(self, dataloader):  # Evaluation for given dataloader
        self.model.load_state_dict(torch.load(self.path))
        self.model.to(self.device).eval()
        for batch_x, _, _ in dataloader:
            y, _, _ = self.model(batch_x.to(self.device, dtype=torch.float))
            y = y.data.cpu().numpy()
        return y, _
