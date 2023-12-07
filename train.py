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

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # input data info
    parser.add_argument('--train_dir', type=str, default=r"C:\Users\kaden\Main\CS678\final_project\sets\ASL")
    parser.add_argument('--val_dir', type=str, default=r"C:\Users\kaden\Main\CS678\final_project\sets\0.1_test")
    parser.add_argument('--in_channels', type=int, default=3)
    parser.add_argument('--num_classes', type=int, default=36)

    # hyperparameters
    parser.add_argument('--batchsize', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=15)

    # other
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda')

    # save directories
    parser.add_argument('--save_dir', type=str, default=r"C:\Users\kaden\Main\CS678\final_project\tests\90-10_alldata_bs32_lr0.01")

    args = parser.parse_args()

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize(224, antialias=True)
    ])

    train_dataset = ImageFolder(args.train_dir, transform)
    train_dataloader = DataLoader(train_dataset, args.batchsize, shuffle=True, num_workers=args.num_workers)

    val_dataset = ImageFolder(args.val_dir, transform)
    val_dataloader = DataLoader(val_dataset, args.batchsize, shuffle=False, num_workers=args.num_workers)

    model = MobileNetV2(args.in_channels, args.num_classes)
    model.to(args.device)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()

    train_l = []
    val_l = []
    val_acc = []
    for epoch in range(0, args.epochs):
        print(f'\n Epoch: {epoch}')

        train_loss = torch.zeros(1, device=args.device)
        model.train()

        # Train model
        for data in tqdm(train_dataloader, desc='Training MobileNetV2...'):

            optimizer.zero_grad()
            model.zero_grad()

            in_data = data[0].cuda()
            labels = data[1].cuda()

            out = model(in_data)

            loss = criterion(out, labels)

            train_loss += loss

            loss.backward()
            optimizer.step()

        # print(f'\ntrain_loss: {train_loss.detach().cpu().numpy()}')
        train_l.append(train_loss.detach().cpu().numpy())

        # Test model on validation set
        with torch.no_grad():
            model.eval()
            val_loss = torch.zeros(1, device=args.device)
            num_correct = torch.zeros(1, device=args.device)
            true_label = []
            pred_label = []
            for data in tqdm(val_dataloader, desc='Testing MobileNetV2...'):
                in_data = data[0].cuda()
                labels = data[1].cuda()

                for label in labels:
                    true_label.append(label.detach().cpu().numpy())

                out = model(in_data)

                loss = criterion(out, labels)

                val_loss += loss

                loc = torch.where(torch.argmax(out, dim=1) == labels, 1, 0)
                num_correct += torch.sum(loc)

                preds = torch.argmax(out, dim=1)
                for pred in preds:
                    pred_label.append(pred.detach().cpu().numpy())

            print(f'\nval_loss: {val_loss.detach().cpu().numpy()}, num_correct: {num_correct.detach().cpu().numpy() / len(val_dataset)}')
            val_l.append(val_loss.detach().cpu().numpy())
            val_accuracy = num_correct.detach().cpu().numpy() / len(val_dataset)
            val_acc.append(val_accuracy)

        if os.path.isdir(args.save_dir):
            # Calcualte F1 score
            fscore = f1_score(true_label, pred_label, average='micro', pos_label=None)

            # Make folders for save information
            cm_save = os.path.join(args.save_dir, 'cm')
            weights_save = os.path.join(args.save_dir, 'weights')
            os.makedirs(weights_save, exist_ok=True)
            os.makedirs(cm_save, exist_ok=True)

            # Save parameters of current model
            torch.save({'epoch': epoch, 'model': model.state_dict(), 'optim': optimizer.state_dict},
                       os.path.join(weights_save, f'weights_{epoch}.pth'))

            # Compute and save confusion matrix
            confusion_matrix = metrics.confusion_matrix(true_label, pred_label, normalize='true')
            cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix,
                                                        display_labels=['0', '1', '2', '3', '4', '5', '6', '7', '8',
                                                                        '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',
                                                                        'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q',
                                                                        'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'])
            fig, ax = plt.subplots(figsize=(25, 25))
            cm_display.plot(ax=ax)
            plt.savefig(os.path.join(cm_save, f'confusion_matrix_{epoch}.jpg'))
            plt.close()

            file_dir = os.path.join(args.save_dir, 'stats.txt')
            if os.path.isfile(file_dir):
                f = open(file_dir, 'a')
            else:
                f = open(file_dir, 'w')

            f.write(f'train_loss: {train_loss.detach().cpu().numpy().item()}, '
                    f'val_loss: {val_loss.detach().cpu().numpy().item()}, val_acc: {val_accuracy.item()},'
                    f' f1_score: {fscore}\n\n')
            f.flush()






