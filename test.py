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

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # input data info
    parser.add_argument('--test_dir', type=str, default=r"C:\Users\kaden\Downloads\ASL_digits\ASL Digits\test")
    parser.add_argument('--weights_dir', type=str, default=r"C:\Users\kaden\Main\CS678\final_project\1\weights\weights_1.pth")
    parser.add_argument('--in_channels', type=int, default=3)
    parser.add_argument('--num_classes', type=int, default=10)

    # hyperparameters
    parser.add_argument('--batchsize', type=int, default=16)

    # other
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda')

    # save directories
    parser.add_argument('--save_dir', type=str, default=r"C:\Users\kaden\Main\CS678\final_project\1")

    args = parser.parse_args()

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize(224, antialias=True)
    ])

    test_dataset = ImageFolder(args.test_dir, transform)
    test_dataloader = DataLoader(test_dataset, args.batchsize, shuffle=True, num_workers=args.num_workers)

    model = MobileNetV2(args.in_channels, args.num_classes)

    checkpoint = torch.load(args.weights_dir)
    model.load_state_dict(checkpoint['model'])
    print('Weights Loaded\n')

    model.to(args.device)

    criterion = torch.nn.CrossEntropyLoss()

    # Test model on testn set
    with torch.no_grad():
        model.eval()
        test_loss = torch.zeros(1, device=args.device)
        num_correct = torch.zeros(1, device=args.device)
        true_label = []
        pred_label = []
        test_l = []
        test_acc = []
        for data in tqdm(test_dataloader, desc='Testing MobileNetV2...'):
            in_data = data[0].cuda()
            labels = data[1].cuda()

            for label in labels:
                true_label.append(label.detach().cpu().numpy())

            out = model(in_data)

            loss = criterion(out, labels)

            test_loss += loss

            loc = torch.where(torch.argmax(out, dim=1) == labels, 1, 0)
            num_correct += torch.sum(loc)

            preds = torch.argmax(out, dim=1)
            for pred in preds:
                pred_label.append(pred.detach().cpu().numpy())

        # print(f'\nval_loss: {val_loss.detach().cpu().numpy()}, num_correct: {num_correct.detach().cpu().numpy() / len(val_dataset)}')
        test_l.append(test_loss.detach().cpu().numpy())
        test_accuracy = num_correct.detach().cpu().numpy() / len(test_dataset)
        test_acc.append(test_accuracy)

        if os.path.isdir(args.save_dir):
            # Compute and save confusion matrix
            confusion_matrix = metrics.confusion_matrix(true_label, pred_label)
            cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix,
                                                        display_labels=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
            cm_display.plot()
            plt.savefig(os.path.join(args.save_dir, f'confusion_matrix.jpg'))

            print(f'test_loss: {test_loss.detach().cpu().numpy().item()}, test_acc: {test_accuracy.item()}\n')
