import torch
from tqdm import tqdm


def train_model(dataloader, model, criterion, optimizer, device='cuda', metric=None, loss_metric=None):
    # get the weight for accumulating the loss
    weight = len(dataloader.dataset)
    train_loss = None

    if loss_metric is None:
        train_loss = torch.zeros(1, device=device)

    # Set to train mode
    model.train()

    # Run the training on a minibatch
    for batch in tqdm(dataloader, desc='Training MobileNetV2...'):
        model.zero_grad(set_to_none=True)

        # get 'em in dat dur gee pee ew
        in_data = batch[0].to(device)
        labels = batch[1].to(device)

        with torch.autocast(device_type=device):
            out = model(in_data)
            loss = criterion(out, labels)

        if loss_metric:
            loss_metric.update(loss.detach(), weight=weight)
        else:
            train_loss += loss

        if metric is not None:
            for m in metric:
                m.update(out, labels)

        loss.backward()
        optimizer.step()

    if loss_metric is None:
        train_loss = train_loss.detach().cpu().numpy()

    return train_loss, loss_metric, metric


def evaluate_model(dataloader, model, criterion, device='cuda', metric=None, loss_metric=None):
    # weight for avg loss
    weight = len(dataloader.dataset)
    val_loss = None

    with torch.no_grad():

        # set to test mode
        model.eval()
        if loss_metric is None:
            val_loss = torch.zeros(1, device=device)

        with torch.autocast(device_type="cuda"):
            for batch in tqdm(dataloader, desc='Testing MobileNetV2...'):

                in_data = batch[0].to(device)
                labels = batch[1].to(device)

                out = model(in_data)

                loss = criterion(out, labels)

                if loss_metric is None:
                    val_loss += loss
                else:
                    loss_metric.update(loss.detach(), weight=weight)

                if metric is not None:
                    for m in metric:
                        m.update(out, labels)

        # make sure not to indent this return statement like an idiot!
        return val_loss, loss_metric, metric
