import torch, os
import torch.utils.data
import numpy as np
from torch import nn, optim
from torch.nn import functional as F


from sklearn.metrics import roc_auc_score


class FAD_class:
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
        self.model = FAD_class.FairClass(
            input_size, num_layers_z, num_layers_y, step_z, step_y
        )
        self.model.to(self.device)
        self.path = os.path.join(
            save_dir,
            name,
        )

    class FairClass(nn.Module):
        def __init__(self, inp_size, num_layers_z, num_layers_y, step_z, step_y):
            super(FAD_class.FairClass, self).__init__()

            num_layers_A = num_layers_y
            lst_z = nn.ModuleList()
            lst_1 = nn.ModuleList()
            lst_2 = nn.ModuleList()
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

            self.fc1 = nn.Sequential(*lst_z)
            self.fc2 = nn.Sequential(*lst_1)
            self.fc3 = nn.Sequential(*lst_2)

        def forward(self, x):
            z = self.fc1(x)
            y = torch.sigmoid(self.fc2(z))
            A = torch.sigmoid(self.fc3(z))
            return y, A

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

        self.model.train()
        nll_criterion = F.binary_cross_entropy
        list_1 = list(self.model.fc1.parameters()) + list(self.model.fc2.parameters())
        list_2 = list(self.model.fc3.parameters())
        optimizer_1 = torch.optim.Adam(list_1, lr=learning_rate)
        optimizer_2 = torch.optim.Adam(list_2, lr=learning_rate)

        prev_loss_y = 9e10
        no_val = 0

        for e in range(max_epoch):
            for batch_x, batch_y, batch_A in dataloader:

                batch_x = batch_x.to(self.device, dtype=torch.float)
                batch_y = batch_y.unsqueeze(dim=1).to(self.device, dtype=torch.float)
                batch_A = batch_A.unsqueeze(dim=1).to(self.device, dtype=torch.float)

                output_1, output_2 = self.model(batch_x)

                loss2 = nll_criterion(output_2, batch_A)
                optimizer_2.zero_grad()
                loss2.backward()
                optimizer_2.step()

                output_1, output_2 = self.model(batch_x)
                loss1 = nll_criterion(output_1, batch_y) - alpha * nll_criterion(
                    output_2, batch_A
                )

                optimizer_1.zero_grad()
                loss1.backward()
                optimizer_1.step()

            if e % log_epoch == 0 and log == 1:

                self.model.eval()
                for x_val, y_val, A_val in dataloader_val:

                    x_val = x_val.to(self.device, dtype=torch.float)
                    y_val = y_val.unsqueeze(dim=1).to(self.device, dtype=torch.float)
                    A_val = A_val.unsqueeze(dim=1).to(self.device, dtype=torch.float)

                    out_1_val, out_2_val = self.model(x_val)

                    loss_y_val = (
                        (
                            nll_criterion(out_1_val, y_val)
                            - alpha * nll_criterion(output_2, batch_A)
                        )
                        .data.cpu()
                        .numpy()
                    )
                    loss_A_val = nll_criterion(out_2_val, A_val).data.cpu().numpy()

                    if loss_y_val > prev_loss_y:
                        no_val += 1
                    else:
                        prev_loss_y, _ = loss_y_val, loss_A_val
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
        self.model.to(self.device).eval()
        self.model.load_state_dict(torch.load(self.path))
        for batch_x, _, _ in dataloader:
            y, A = self.model(batch_x.to(self.device, dtype=torch.float))
            y, A = y.data.cpu().numpy(), A.data.cpu().numpy()
        return y, _
