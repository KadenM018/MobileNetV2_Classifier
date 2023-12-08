import argparse
import os
import torchvision
import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from MobileNetV2 import MobileNetV2
from tqdm import tqdm
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from dataparse import add_args, create_dataframe, ASLDataset
import Trainer
from torcheval.metrics.classification import MulticlassF1Score, MulticlassConfusionMatrix
from torcheval.metrics import Mean

def main():
    # parse input from terminal
    parser = argparse.ArgumentParser()
    args = add_args(parser)

    # transforms for training data
    transform_train = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize(224, antialias=True),
        torchvision.transforms.RandomHorizontalFlip(p=0.2)
    ])

    transform_val = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize(224, antialias=True),
    ])

    # get dataset paths and labels
    df = create_dataframe(args.data_path)

    # get test/train split. If not too slow, we can use stratified K fold instead
    train_split, test_split = train_test_split(df, test_size=0.2, random_state=42)

    # training data
    train_dataset = ASLDataset(train_split, transform_train)
    train_dataloader = DataLoader(train_dataset, args.batchsize, shuffle=True, num_workers=args.num_workers)

    # validation data
    val_dataset = ASLDataset(test_split, transform_val)
    val_dataloader = DataLoader(val_dataset, args.batchsize, shuffle=False, num_workers=args.num_workers)

    # get the model, put it on the gpu
    model = MobileNetV2(args.in_channels, args.num_classes)
    model.to(args.device)

    # select our optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()

    # create metrics
    train_f1_metric = MulticlassF1Score(num_classes=36,device=args.device)
    train_loss_metric = Mean(device=args.device)
    val_f1_metric = MulticlassF1Score(num_classes=36, device=args.device)
    val_loss_metric = Mean(device=args.device)
    val_confusion_metric = MulticlassConfusionMatrix(num_classes=36,device=args.device)

    # prepare for saving stuff
    if os.path.isdir(args.save_dir):
        # Make folders for save information
        cm_save = os.path.join(args.save_dir, 'cm')
        weights_save = os.path.join(args.save_dir, 'weights')
        os.makedirs(weights_save, exist_ok=True)
        os.makedirs(cm_save, exist_ok=True)

    train_f1 = []
    train_loss = []
    val_f1 = []
    val_loss = []
    val_conf = None

    best_loss = 9999999.
    for epoch in range(0, args.epochs):
        print(f'\n Epoch: {epoch}')

        _, train_loss_metric, (train_f1_metric) = (
            Trainer.train_model(train_dataloader, model,
                                device=args.device,
                                optimizer=optimizer,
                                criterion=criterion,
                                loss_metric=train_loss_metric,
                                metric=[train_f1_metric]))

        # Test model on validation set
        _, val_loss_metric, (val_f1_metric, val_confusion_metric) = (
            Trainer.evaluate_model(val_dataloader, model,
                                   device=args.device,
                                   criterion=criterion,
                                   loss_metric=val_loss_metric,
                                   metric=[val_f1_metric,val_confusion_metric]))

        # Compute the epoch metrics
        train_loss.append(train_loss_metric.compute())
        train_f1.append(train_f1_metric.compute())
        val_loss.append(val_loss_metric.compute())
        val_f1.append(val_f1_metric.compute())
        val_conf = val_confusion_metric.compute()

        # Save only the best model by loss
        if val_loss[-1] < best_loss:
            best_loss = val_loss[-1]
            if args.bestonly:
                torch.save({'epoch': epoch, 'model': model.state_dict(), 'optim': optimizer.state_dict},
                           os.path.join(weights_save, f'best_weights_{epoch}.pth'))

        # Save parameters of current model every epoch
        if not args.bestonly:
            torch.save({'epoch': epoch, 'model': model.state_dict(), 'optim': optimizer.state_dict},
                            os.path.join(weights_save, f'weights_{epoch}.pth'))

            # Compute and save confusion matrix
            cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=val_conf.detach().cpu().numpy(),
                                                        display_labels=['0', '1', '2', '3', '4', '5', '6', '7', '8',
                                                                        '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',
                                                                        'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q',
                                                                        'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'])

            fig, ax = plt.subplots(figsize=(25, 25))
            cm_display.plot(ax=ax)
            plt.savefig(os.path.join(cm_save, f'confusion_matrix_{epoch}.jpg'))
            plt.close()

            file_dir = os.path.join(args.save_dir, 'stats.txt')
            with open(file_dir, 'a') as f:
                f.write(f'train_loss: {train_loss.detach().cpu().numpy().item()}, '
                    f'val_loss: {val_loss.detach().cpu().numpy().item()}, val_acc: {val_accuracy.item()},'
                    f' f1_score: {fscore}\n\n')
            f.flush()



if __name__ == "__main__":
    main()
