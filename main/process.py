import torch, os, sys

abs_path = os.path.abspath(os.getcwd())
sys.path.append(os.path.join(abs_path, "src"))

import numpy as np
import models
import pickle


from utilities import Adult_dataset
from utilities import medical_dataset
from utilities import german_dataset_age
from utilities import german_dataset_sex

from utilities import format_datasets
from utilities import Dataset_format
from utilities import test


from torch.utils.data import DataLoader
from xlwt import Workbook


def _process(
    num_layers_z=None,
    num_layers_y=None,
    num_layers_w=None,
    num_layers_A=None,
    step_z=None,
    step_y=None,
    step_A=None,
    step_w=None,
    epochs=3000,
    threshold=0.5,
    model_AIF=[],
    alpha=[0, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000],
    eta=[0, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000],
    saver_dir_models=r"data/Trained_models",
    saver_dir_results=r"data/Results",
    name="Adult",
):

    """INPUT DATA"""

    saver_dir_models = os.path.join(saver_dir_models, name)
    saver_dir_results = os.path.join(saver_dir_results, name)
    os.makedirs(saver_dir_models, exist_ok=True)
    os.makedirs(saver_dir_results, exist_ok=True)

    if name == "Adult":
        data, atribute, sensitive, output, pr_gr, un_gr = Adult_dataset()
    elif name == "Ger_age":
        data, atribute, sensitive, output, pr_gr, un_gr = german_dataset_age()
    elif name == "Ger_sex":
        data, atribute, sensitive, output, pr_gr, un_gr = german_dataset_sex()
    elif name == "Medical_exp":
        data, atribute, sensitive, output, pr_gr, un_gr = medical_dataset()

    prot = list(pr_gr[0].keys())[0]

    (
        data_train,
        data_test,
        data_val,
        atribute_train,
        atribute_val,
        atribute_test,
        sensitive_train,
        sensitive_val,
        sensitive_test,
        output_train,
        output_val,
        output_test,
    ) = format_datasets(data, atribute, sensitive, output, sens_name=prot)

    dataset_train = Dataset_format(atribute_train, sensitive_train, output_train)
    dataset_val = Dataset_format(atribute_val, sensitive_val, output_val)
    dataset_test = Dataset_format(atribute_test, sensitive_test, output_test)

    dataloader_train = DataLoader(
        dataset_train, batch_size=len(dataset_train), shuffle=True
    )
    dataloader_val = DataLoader(dataset_val, batch_size=len(dataset_val), shuffle=False)
    dataloader_test = DataLoader(
        dataset_test, batch_size=len(dataset_test), shuffle=False
    )

    inp = dataset_train[0][0].shape[0]

    iteracija = 0

    wb = Workbook()

    columns = [
        "model",
        "AUC_y_val",
        "Accuracy_y_val",
        "AUC_A_val",
        "bal_acc_val",
        "avg_odds_diff_val",
        "disp_imp_val",
        "stat_par_diff_val",
        "eq_opp_diff_val",
        "AUC_y_test",
        "Accuracy_y_test",
        "AUC_A_test",
        "bal_acc_test",
        "avg_odds_diff_test",
        "disp_imp_test",
        "stat_par_diff_test",
        "eq_opp_diff_test",
        "alpha",
    ]

    sheets = wb.add_sheet("{}".format("sheet_1"))

    k = 0
    for i in columns:
        sheets.write(0, k, i)
        k += 1

    row = 1

    for a in alpha:

        ind = eta[iteracija]
        iteracija += 1

        lst = [
            models.FAD_class(
                input_size=inp,
                num_layers_z=num_layers_z,
                num_layers_y=num_layers_y,
                step_z=step_z,
                step_y=step_y,
                name=f"FAD_{a}",
                save_dir=saver_dir_models,
            ),
            models.FAIR_scalar_class(
                input_size=inp,
                num_layers_w=num_layers_w,
                step_w=step_w,
                num_layers_A=num_layers_A,
                step_A=step_A,
                num_layers_y=num_layers_y,
                step_y=step_y,
                name=f"FAIR_scalar_{a}",
                save_dir=saver_dir_models,
            ),
            models.FAIR_betaSF_class(
                input_size=inp,
                num_layers_w=num_layers_w,
                step_w=step_w,
                num_layers_A=num_layers_A,
                step_A=step_A,
                num_layers_y=num_layers_y,
                step_y=step_y,
                name=f"FAIR_betaSF_{a}",
                save_dir=saver_dir_models,
            ),
            models.FAIR_Bernoulli_class(
                input_size=inp,
                num_layers_w=num_layers_w,
                step_w=step_w,
                num_layers_A=num_layers_A,
                step_A=step_A,
                num_layers_y=num_layers_y,
                step_y=step_y,
                name=f"FAIR_Bernoulli_{a}",
                save_dir=saver_dir_models,
            ),
            models.FAIR_betaREP_class(
                input_size=inp,
                num_layers_w=num_layers_w,
                step_w=step_w,
                num_layers_A=num_layers_A,
                step_A=step_A,
                num_layers_y=num_layers_y,
                step_y=step_y,
                name=f"FAIR_betaREP_{a}",
                save_dir=saver_dir_models,
            ),
            models.CLFR_class(
                input_size=inp,
                num_layers_z=num_layers_z,
                num_layers_y=num_layers_y,
                step_z=step_z,
                step_y=step_y,
                name=f"CLFR_A_{a}",
                save_dir=saver_dir_models,
            ),
            models.LURMI_class(
                input_size=inp,
                num_layers_z=num_layers_z,
                num_layers_y=num_layers_y,
                step_z=step_z,
                step_y=step_y,
                name=f"LURMI_A_{a}",
                save_dir=saver_dir_models,
            ),
        ]

        k = 0
        for i in lst:

            try:
                if np.isin(k, model_AIF):
                    i.fit(data_train, ["labels"], [prot])
                    saver_path = os.path.join(
                        saver_dir_models,
                        "checkpoint_{}_epochs_{}_eta_{}".format(
                            type(i).__name__, epochs, ind
                        ),
                    )
                else:
                    i.fit(
                        dataloader_train,
                        dataloader_val,
                        max_epoch=epochs,
                        log=1,
                        alpha=a,
                        log_epoch=2,
                        early_stopping_no=100,
                    )
                    saver_path = os.path.join(
                        saver_dir_models,
                        "checkpoint_{}_epochs_{}_alpha_{}".format(
                            type(i).__name__, epochs, a
                        ),
                    )

                f = open(saver_path, "wb")
                pickle.dump(i, f)
                f.close

                metric_val, metric_test = test(
                    data_val,
                    data_test,
                    i,
                    output_val,
                    output_test,
                    sensitive_val,
                    sensitive_test,
                    threshold,
                    model_AIF,
                    k,
                    dataloader_val,
                    dataloader_test,
                    prot,
                    un_gr,
                    pr_gr,
                )

                print(metric_val, metric_test)

                for column, _ in enumerate(columns):
                    if column == 0:
                        name = type(i).__name__
                        sheets.write(row, column, name)
                    elif column > 0 and column < 9:
                        sheets.write(row, column, metric_val[column - 1])
                    elif column == len(columns) - 1:
                        sheets.write(row, column, a)
                    else:
                        sheets.write(row, column, metric_test[column - 9])

                wb.save(saver_dir_results)

                row += 1
                k += 1
            except:
                continue
