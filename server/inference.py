import numpy as np
import lightning.pytorch as pl
import torch
import neurokit2 as nk
from resnet1d import ResNet1D


class Lightning_ResNet1D(pl.LightningModule):
    def __init__(self, model_hparams=None, model_class=ResNet1D):
        """
        Inputs:
            model_name - Name of the model/CNN to run. Used for creating the model (see function below)
            model_hparams - Hyperparameters for the model, as dictionary.
            optimizer_name - Name of the optimizer to use. Currently supported: Adam, SGD
            optimizer_hparams - Hyperparameters for the optimizer, as dictionary. This includes learning rate, weight decay, etc.
        """
        super().__init__()
        # Exports the hyperparameters to a YAML file, and create "self.hparams" namespace
        if model_hparams is None:
            model_hparams = {"n_classes": 7, "base_filters": 16, "kernel_size": 16, "stride": 2, "groups": 1,
                             "n_block": 12, "in_channels": 12}
        self.save_hyperparameters()
        # Create model
        self.model = model_class(**model_hparams)
        # Create loss module

        # Example input for visualizing the graph in Tensorboard
        self.example_input_array = torch.zeros((1, 12, 500), dtype=torch.float32)

    def forward(self, imgs):
        # Forward function that is run when visualizing the graph
        return self.model(imgs)

    def training_step(self, batch, batch_idx):
        # "batch" is the output of the training data loader.
        self.model.train()
        imgs, labels = batch
        labels = np.squeeze(labels)
        preds = np.squeeze(self.model(imgs))
        loss = self.loss_module(preds, labels)

        self.train_acc(preds, labels.to(torch.int))

        self.train_score(preds, labels.to(torch.int))
        self.log("train_f1_score", self.train_score)
        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        self.model.eval()
        imgs, labels = batch
        labels = np.squeeze(labels)
        preds = np.squeeze(self.model(imgs))

        self.val_acc(preds, labels.to(torch.int))
        self.val_score(preds, labels.to(torch.int))

        self.log("val_f1_score", self.val_score)
        self.log("val_acc", self.val_acc)

    def predict_step(self, batch, batch_idx):
        preds = np.squeeze(self(batch))
        return preds


class Transformator():
    def __init__(self, args, transformation_func, is_train):
        self.is_train = is_train
        self.args = args
        self.transformation_func = transformation_func

    # transformation func - function that transforms record to any 
    def pipeline_record(self, transformation_func, record_name):
        path = "./transformed_inference/"

        new_names_column = []
        transformed = transformation_func(record_name, **self.args)
        # transformed - shape [12, 9] if preprocessing_with beats

        names = []
        for i in range(transformed.shape[1]):
            name = f"{record_name}_transformed_{i}"
            names.append(name)
            np.save(path + name + ".npy", transformed[:, i])

        return names


def predict(test_beat, resnet_model):
    test_beat = test_beat.reshape((1, 12, -1))
    res = resnet_model(torch.from_numpy(test_beat).float())
    return res.detach().numpy()


def inference_model(record_pth, resnet_model):
    ecg_signal = np.load(record_pth)

    # Automatically process the (raw) ECG signal
    signals, info = nk.ecg_process(ecg_signal[0], sampling_rate=500)

    # Extract clean ECG and R-peaks location
    cleaned_ecg = signals["ECG_Clean"]

    # Spotting all the heart beats
    epochs = nk.ecg_segment(cleaned_ecg, rpeaks=None, sampling_rate=500, show=False)

    model_predicts = []
    for i in list(epochs.keys())[1:-1]:
        record_ecg_signal = ecg_signal[1:, epochs[i].Index.min():epochs[i].Index.max() + 1]
        record_ecg_signal = record_ecg_signal[:, :500]
        record_ecg_signal = np.pad(record_ecg_signal, ((0, 0), (0, 500 - len(record_ecg_signal[0]))), 'constant',
                                   constant_values=0)

        cleaned_ecg = epochs[i].Signal.to_numpy()[:500]
        cleaned_ecg = np.pad(cleaned_ecg, (0, 500 - len(cleaned_ecg)), 'constant', constant_values=0)

        record_ecg_signal = np.insert(record_ecg_signal, 0, cleaned_ecg, axis=0)

        model_predicts.append(predict(record_ecg_signal, resnet_model)[0])

    return np.array(model_predicts).mean(axis=0)

# if __name__ == "__main__":
# record_pth = "00127_hr.npy"
# resnet_model = Lightning_ResNet1D.load_from_checkpoint("model.ckpt",
#                                                        model_hparams={"n_classes": 7, "base_filters": 16,
#                                                                       "kernel_size": 16, "stride": 2, "groups": 1,
#                                                                       "n_block": 12, "in_channels": 12},
#                                                        map_location=torch.device('cpu'))
# preds = inference_model(record_pth, resnet_model)
