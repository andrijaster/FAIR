import torch, os
import torch.utils.data
import numpy as np
from torch import nn
from torch.nn import functional as F


class FAIR_scalar_class:
    class Fair_classifier(nn.Module):
        def __init__(
            self,
            inp_size,
            num_layers_w,
            step_w,
            num_layers_A,
            step_A,
            num_layers_y,
            step_y,
        ):
            super(FAIR_scalar_class.Fair_classifier, self).__init__()

            lst_z = nn.ModuleList()
            lst_A = nn.ModuleList()
            lst_y = nn.ModuleList()
            out_size_A = inp_size
            out_size_y = inp_size
            out_size = inp_size

            for i in range(num_layers_w):
                inp_size = out_size
                out_size = int(inp_size // step_w)
                if i == num_layers_w - 1:
                    block = nn.Linear(inp_size, 1)
                else:
                    block = nn.Sequential(
                        nn.Linear(inp_size, out_size),
                        nn.BatchNorm1d(num_features=out_size, momentum=0.01),
                        nn.ReLU(),
                    )
                lst_z.append(block)

            for i in range(num_layers_A):
                inp_size = out_size_A
                out_size_A = int(inp_size // step_A)
                if i == num_layers_A - 1:
                    block = nn.Linear(inp_size, 1)
                else:
                    block = nn.Sequential(
                        nn.Linear(inp_size, out_size_A),
                        nn.BatchNorm1d(num_features=out_size_A, momentum=0.01),
                        nn.ReLU(),
                    )
                lst_A.append(block)

            for i in range(num_layers_y):
                inp_size = out_size_y
                out_size_y = int(inp_size // step_y)
                if i == num_layers_y - 1:
                    block = nn.Linear(inp_size, 1)
                else:
                    block = nn.Sequential(
                        nn.Linear(inp_size, out_size_y),
                        nn.BatchNorm1d(num_features=out_size_y, momentum=0.01),
                        nn.ReLU(),
                    )
                lst_y.append(block)

            self.fc1 = nn.Sequential(*lst_y)

            self.fc2 = nn.Sequential(*lst_A)

            self.fc3 = nn.Sequential(*lst_z)

        def forward(self, x):
            output_y = torch.sigmoid(self.fc1(x))
            output_w = torch.sigmoid(self.fc3(x))
            output_A = torch.sigmoid(self.fc2(x))
            return output_y, output_A, output_w

    def __init__(
        self,
        input_size,  # input size
        num_layers_w,  # no. layers in weighting network
        step_w,  # step in weighting network
        num_layers_A,  # no. layers in sensitive network
        step_A,  # step in sensitive network
        num_layers_y,  # no. layers in output network
        step_y,  # step in output network
        name="Fair_scalar",
        save_dir=None,
    ):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = FAIR_scalar_class.Fair_classifier(
            input_size, num_layers_w, step_w, num_layers_A, step_A, num_layers_y, step_y
        )
        self.model.to(self.device)
        self.path = os.path.join(save_dir, name)

    def fit(
        self,
        dataloader,  # train dataloader
        dataloader_val,  # validation dataloader
        early_stopping_no=3,  # early stopping no.
        max_epoch=300,  # max_epochs
        alpha=1.0,  # hyperparameter \alpha
        beta=0,  # regularization hyperparameter
        log_epoch=1,  # no. of epoch for evaluation of validation loss
        log=0,  # evaluate validation loss {1 - Yes, 0 - No}
        learning_rate=0.0001,  # learning rate
    ):
        def loss(output, target, weights):
            output = torch.clamp(output, 1e-5, 1 - 1e-5)
            weights = torch.clamp(weights, 1e-5, 1 - 1e-5)
            ML = weights * (
                target * torch.log(output) + (1 - target) * torch.log(1 - output)
            )
            return torch.neg(torch.mean(ML))

        nll_criterion = F.binary_cross_entropy
        list_0 = list(self.model.fc1.parameters())
        list_1 = list(self.model.fc3.parameters())
        list_2 = list(self.model.fc2.parameters())

        optimizer_0 = torch.optim.Adam(list_0, lr=learning_rate)
        optimizer_1 = torch.optim.Adam(list_1, lr=learning_rate)
        optimizer_2 = torch.optim.Adam(list_2, lr=learning_rate)

        prev_loss_y, prev_loss_A = 9e10, 9e10

        for e in range(max_epoch):
            for batch_x, batch_y, batch_A in dataloader:
                self.model.train()

                batch_x = batch_x.to(self.device, dtype=torch.float)
                batch_y = batch_y.unsqueeze(dim=1).to(self.device, dtype=torch.float)
                batch_A = batch_A.unsqueeze(dim=1).to(self.device, dtype=torch.float)

                y, A, w = self.model(batch_x)
                loss0 = loss(y, batch_y, w)
                optimizer_0.zero_grad()
                loss0.backward()
                optimizer_0.step()

                y, A, w = self.model(batch_x)
                loss2 = loss(A, batch_A, w)
                optimizer_2.zero_grad()
                loss2.backward()
                optimizer_2.step()

                y, A, w = self.model(batch_x)
                loss1 = (
                    loss(y, batch_y, w)
                    - alpha * loss(A, batch_A, w)
                    - beta * torch.norm(w, 1)
                )
                optimizer_1.zero_grad()
                loss1.backward()
                optimizer_1.step()

            if e % log_epoch == 0 and log == 1:

                for x_val, y_val, A_val in dataloader_val:

                    x_val = x_val.to(self.device, dtype=torch.float)
                    y_val = y_val.unsqueeze(dim=1).to(self.device, dtype=torch.float)
                    A_val = A_val.unsqueeze(dim=1).to(self.device, dtype=torch.float)

                    out_1_val, out_2_val, _ = self.model(x_val)

                    loss_y_val = nll_criterion(out_1_val, y_val).data.cpu().numpy()
                    loss_A_val = nll_criterion(out_2_val, A_val).data.cpu().numpy()

                    if loss_y_val > prev_loss_y:
                        no_val += 1
                    else:
                        prev_loss_y, prev_loss_A = loss_y_val, loss_A_val
                        torch.save(self.model.state_dict(), self.path)
                        print("Model saved")
                        no_val = 0

                if no_val == early_stopping_no:
                    break

    def predict(self, x_test):  # Inference
        y, A, w = self.model(x_test)
        y = np.round(y.data)
        A = np.round(A.data)
        w = w.data
        return y, A

    def predict_proba(self, dataloader):  # Evaluation for given dataloader
        for x_test, _, _ in dataloader:
            y, A, w = self.model(x_test.to(self.device, dtype=torch.float))
            y = y.data.cpu().numpy()
            A = A.data.cpu().numpy()
            w = w.data.cpu().numpy()
        return y, _
