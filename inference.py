import pandas as pd
import numpy as np
import lightning.pytorch as pl
import torch
import argparse

class Lightning_ResNet1D(pl.LightningModule):
    def __init__(self, model_name, model_hparams, optimizer_name, optimizer_hparams, model_class=ResNet1D, task="MULTILABEL"):
        """
        Inputs:
            model_name - Name of the model/CNN to run. Used for creating the model (see function below)
            model_hparams - Hyperparameters for the model, as dictionary.
            optimizer_name - Name of the optimizer to use. Currently supported: Adam, SGD
            optimizer_hparams - Hyperparameters for the optimizer, as dictionary. This includes learning rate, weight decay, etc.
        """
        super().__init__()
        # Exports the hyperparameters to a YAML file, and create "self.hparams" namespace
        self.save_hyperparameters()
        # Create model
        self.model = model_class(**model_hparams)
        # Create loss module
        self.loss_module = nn.CrossEntropyLoss()
        self.train_score = F1Score(task=task, num_classes=model_hparams["n_classes"], top_k=1)
        self.val_score = MulticlassF1Score(task=task, num_classes=model_hparams["n_classes"], top_k=1)
        self.test_score = F1Score(task=task, num_labels=model_hparams["n_classes"], num_classes=model_hparams["n_classes"], top_k=1)
        self.val_acc = Accuracy(task=task, num_classes=model_hparams["n_classes"], top_k=1)
        self.train_acc = Accuracy(task=task, num_labels=model_hparams["n_classes"], num_classes=model_hparams["n_classes"], top_k=1)
        

        # Example input for visualizing the graph in Tensorboard
        self.example_input_array = torch.zeros((1, 12, 500), dtype=torch.float32)

    def forward(self, imgs):
        # Forward function that is run when visualizing the graph
        return self.model(imgs)

    def configure_optimizers(self):
        # We will support Adam or SGD as optimizers.
        if self.hparams.optimizer_name == "Adam":
            # AdamW is Adam with a correct implementation of weight decay (see here
            # for details: https://arxiv.org/pdf/1711.05101.pdf)
            optimizer = optim.AdamW(self.parameters(), **self.hparams.optimizer_hparams)
        elif self.hparams.optimizer_name == "SGD":
            optimizer = optim.SGD(self.parameters(), **self.hparams.optimizer_hparams)
        else:
            assert False, f'Unknown optimizer: "{self.hparams.optimizer_name}"'

        # We will reduce the learning rate by 0.1 every milestone
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[35,65, 115, 150], gamma=0.1)
        return [optimizer], [scheduler]

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
    


def main():
    parser = argparse.ArgumentParser(description="Read and process user data")
    
    parser.add_argument('--age', type=int, help="User's age")
    parser.add_argument('--sex', choices=['male', 'female'], help="User's sex (male or female)")
    parser.add_argument('--height', type=float, help="User's height in centimeters")
    parser.add_argument('--weight', type=float, help="User's weight in kilograms")
    parser.add_argument('--path', help="Path to the record file")
    
    args = parser.parse_args()
    
    # Check if all required arguments are provided
    if not all(vars(args).values()):
        parser.error("All arguments (age, sex, height, weight, and path) are required.")

    age = args.age
    sex = args.sex
    height = args.height
    weight = args.weight
    record_path = args.path

    # Your code to process the arguments goes here
    # You can perform any desired operations using the provided arguments.

    print(f"Age: {age}")
    print(f"Sex: {sex}")
    print(f"Height: {height} cm")
    print(f"Weight: {weight} kg")
    print(f"Record Path: {record_path}")

if __name__ == "__main__":
    main()
