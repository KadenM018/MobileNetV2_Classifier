import argparse
import os
import torchvision
import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
from MobileNetV2 import MobileNetV2
from tqdm import tqdm
from sklearn import metrics
import matplotlib.pyplot as plt
from dataparse import add_args, ASLDataset
from torcheval.metrics.classification import MulticlassF1Score, MulticlassConfusionMatrix
from torcheval.metrics import Mean

if __name__ == '__main__':

    # parser
    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args()

    # transforms
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize(224, antialias=True),
        torchvision.transforms.CenterCrop(224)
    ])

    ####################### Dataset Loading #######################
    test_dataset = ImageFolder(args.data_path)
    # not sure if I need this since we take all the images, but to be consistent
    test_sub = Subset(test_dataset, [i for i in range(len(test_dataset))])
    testASL = ASLDataset(test_sub, transform=transform)
    test_dataloader = DataLoader(testASL, args.batchsize, shuffle=True, num_workers=args.num_workers)
    ################################################################

    ## Metrics ##
    f1_metric = MulticlassF1Score(num_classes=36, device=args.device)
    loss_metric = Mean(device=args.device)
    confusion_metric = MulticlassConfusionMatrix(num_classes=36, device=args.device)

    ## Load Model ##
    model = MobileNetV2(args.in_channels, args.num_classes)

    checkpoint = torch.load(args.weights_dir)
    model.load_state_dict(checkpoint['model'])
    print('Weights Loaded\n')

    model.to(args.device)

    criterion = torch.nn.CrossEntropyLoss()

    # Test model on test set
    with torch.no_grad():
        model.eval()

        with torch.autocast(device_type="cuda"):
            for batch in tqdm(test_dataloader, desc='Testing MobileNetV2...'):

                in_data = batch[0].to(args.device)
                labels = batch[1].to(args.device)

                out = model(in_data)

                loss = criterion(out, labels)

                loss_metric.update(loss.detach(), weight=len(test_dataloader.dataset))
                f1_metric.update(out, labels)
                confusion_metric.update(out, labels)

        final_loss = loss_metric.compute()
        final_f1 = f1_metric.compute()
        final_conf = confusion_metric.compute()

        if os.path.isdir(args.save_dir):
            # Compute and save confusion matrix
            cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=final_conf.detach().cpu().numpy(),
                                                        display_labels=['0', '1', '2', '3', '4', '5', '6', '7', '8',
                                                                        '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',
                                                                        'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q',
                                                                        'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'])
            cm_display.plot()
            plt.savefig(os.path.join(args.save_dir, "{}_conf_matrix.jpg".format(args.name)))

            print("stats for test {}:\n".format(args.name))
            print(f'loss: {final_loss.detach().cpu().numpy()}, f1: {final_f1.detach().cpu().numpy()}\n')
