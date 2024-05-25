import datetime
import os

import numpy as np
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms

from dataset import DirConverStegoDataset
from model import ViT
from visualization_functions import Plt_hist, build_confusion_matrix, visualize_confusion_matrix

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: ", device, f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "")

algorithm = 'HUGO'

amount_of_pictures = 200000
data_size = 160000

transform = transforms.Compose(
    [
        # transforms.RandomRotation(180, interpolation=PIL.Image.BILINEAR),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor()
    ])
data_train = DirConverStegoDataset(f'./NewMoreImagesDataset/{algorithm}-256cropped/', data_size,
                                   transform=transform)
data_test = DirConverStegoDataset(f'./NewMoreImagesDataset/{algorithm}-256cropped/', amount_of_pictures - data_size,
                                  transform=transform)
# SEFAR10
# data_train = torchvision.datasets.CIFAR10(root='./cifar10', train=True, download=True, transform=transform)
# data_test = torchvision.datasets.CIFAR10(root='./cifar10', train=False, download=True, transform=transform)

batch_size = 32
validation_split = .2
num_epochs = 150
pretrained_epoch = 0
save = True

split = int(np.floor(validation_split * data_size))
indices = list(range(amount_of_pictures))

np.random.shuffle(indices)

train_indices, val_indices, test_indices = indices[split:data_size], indices[:split], indices[data_size:]

train_sampler = SubsetRandomSampler(train_indices)
val_sampler = SubsetRandomSampler(val_indices)
test_sampler = SubsetRandomSampler(test_indices)

train_loader = DataLoader(data_train, batch_size=batch_size, sampler=train_sampler)
val_loader = DataLoader(data_train, batch_size=batch_size, sampler=val_sampler)
test_loader = DataLoader(data_test, batch_size=batch_size, sampler=test_sampler)

# SEFAR10
# train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True)
# val_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True)

# TODO: Слева - с SRM
# TODO: Справа - без SRM

nn_model = ViT(
    image_size=256,
    patch_size=16,
    num_classes=2,
    channels=3,
    dim_model=16,
    depth=1,
    heads=8,
    mlp_dim=32,
    dropout=0.1,  # 0.1
    emb_dropout=0.1,  # 0.1
    device=device
)

# nn_model = VisionTransformer(
#     img_size=256,
#     emb_size=256,
#     device=device
# )
# KovViT(
#     device=device,
#     img_size=256,
#     patch_size=16,
#     n_channels=1,
#     hidden_dim=512,
#     nhead=16,
#     num_layers=4,
#     mlp_dim=1024,
#     n_classes=2,
# dropout = 0.1,
# emb_dropout = 0.1
# )

nn_model.type(torch.cuda.FloatTensor)
nn_model.to(device)

# nn_model.load_state_dict(torch.load(f'./best_checkpoints/cnn_epoch045_HUGO_2024-03-20.pth'))
# Продолжить тренировку с данными из файла
if pretrained_epoch != 0:
    nn_model.load_state_dict(torch.load(f'./trained/cnn_epoch{(pretrained_epoch):03}.pth'))

criterion = nn.CrossEntropyLoss(
    label_smoothing=0.1  # 0.1
).type(torch.cuda.FloatTensor)

optimizer = torch.optim.AdamW(
    nn_model.parameters(),
    lr=0.0005,
    betas=(0.9, 0.999),
    eps=1e-8,
    # weight_decay=0.9,
    weight_decay=0,
)

scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.75, patience=5, min_lr=0.000001)


def train_model(model, train_loader, val_loader, loss, optimizer, scheduler, num_epochs, writer, first_epoch=0):
    loss_history = []
    train_history = []
    val_history = []
    val_losses = []

    best_epoch = -1
    best_epoch_accuracy = 0
    for epoch in range(first_epoch, num_epochs):
        model.train()
        loss_accum = 0
        correct_samples = 0
        total_samples = 0
        running_loss = 0.0
        for i, data in enumerate(train_loader):
            inputs, labels = data
            x_gpu = inputs.to(device)
            y_gpu = labels.to(device)
            optimizer.zero_grad()
            outputs = model(x_gpu)

            if isinstance(outputs, list):
                outputs = outputs[0]

            loss = criterion(outputs, y_gpu)

            loss.backward()
            optimizer.step()

            _, indices = torch.max(outputs, 1)
            correct_samples += torch.sum(indices == y_gpu)
            total_samples += labels.shape[0]

            score = torch.sum(indices == y_gpu) / labels.shape[0]

            loss_accum += loss

            if (i + 1) % 25 == 0:
                print(
                    "epoch {:3d} | iteration {:5d} | Loss {:.6f} | Acc {:.2f}%".format(epoch + 1, i + 1, loss,
                                                                                       score * 100))

        ave_loss = loss_accum / i
        train_accuracy = float(correct_samples) / total_samples
        val_accuracy, val_loss = compute_accuracy(model, val_loader)
        scheduler.step(val_accuracy)

        if val_accuracy > best_epoch_accuracy:
            best_epoch_accuracy = val_accuracy
            best_epoch = epoch + 1

        loss_history.append(float(ave_loss))
        train_history.append(train_accuracy)

        val_history.append(val_accuracy)
        val_losses.append(val_loss)

        writer.add_scalars('Loss', {'train': float(ave_loss)}, epoch + 1)
        writer.add_scalars('Accuracy', {'train': train_accuracy}, epoch + 1)
        writer.add_scalars('Loss', {'validation': val_loss}, epoch + 1)
        writer.add_scalars('Accuracy', {'validation': val_accuracy}, epoch + 1)

        print("Epoch: %d, Average loss: %f, Validation loss: %f, Train accuracy: %f, Val accuracy: %f" % (
            epoch + 1, ave_loss, val_loss, train_accuracy, val_accuracy))

        if save:
            # TODO: trained - c SRM
            # TODO: trained2 - без SRM
            torch.save(model.state_dict(), os.path.join('./trained', "cnn_epoch{:03d}.pth".format(epoch + 1)))
            if (best_epoch == epoch + 1):
                # TODO: best_checkpoints - c SRM
                # TODO: best_checkpoints2 - без SRM
                torch.save(model.state_dict(), os.path.join('./best_checkpoints',
                                                            f"cnn_epoch{epoch + 1:03d}_{algorithm}_{datetime.date.today()}.pth"))
            print("Saving Model of Epoch {}".format(epoch + 1))

    return loss_history, train_history, val_history, val_losses


def compute_accuracy(model, loader):
    model.eval()
    total = 0
    correct = 0
    losses = []
    with torch.no_grad():
        for i_step, (x, y) in enumerate(loader):
            x_gpu = x.to(device)
            y_gpu = y.to(device)
            predictions = model(x_gpu)
            if isinstance(predictions, list):
                predictions = predictions[0]
            loss = criterion(predictions, y_gpu)
            losses.append(loss.item())
            _, predicted = torch.max(predictions.data, 1)
            total += y_gpu.size(0)
            correct += (predicted == y_gpu).sum().item()
    return correct / total, sum(losses) / len(losses)


# TODO: runs - c SRM
# TODO: runs2 - без SRM
writer = SummaryWriter("runs/{}_attention_{:%Y-%m-%d_%H-%M-%S}".format(algorithm, datetime.datetime.now()))
loss_history, train_history, val_history, val_losses = train_model(nn_model, train_loader, val_loader, criterion,
                                                                   optimizer, scheduler, num_epochs, writer,
                                                                   pretrained_epoch)
Plt_hist(loss_history, train_history, val_history, val_losses, writer)

# visualize(nn_model, test_loader, writer, device, batch_size, algorithm)
test_accuracy, test_loss = compute_accuracy(nn_model, test_loader)

print(f"Algorithm: {algorithm} Test accuracy: {test_accuracy}, Test loss: {test_loss}")


def evaluate_model(model, loader, indices):
    model.eval()

    predictions = []
    ground_truth = []
    for index, (x, y) in enumerate(loader):
        x_gpu = x.to(device)
        #         y_gpu=y.to(device)
        prediction = model(x_gpu)

        if isinstance(prediction, list):
            prediction = prediction[0]

        _, prediction = torch.max(prediction.data, 1)
        predictions.extend(prediction)
        ground_truth.extend(y)
    return predictions, ground_truth


predictions, gt = evaluate_model(nn_model, test_loader, test_indices)
confusion_matrix = build_confusion_matrix(predictions, gt)
visualize_confusion_matrix(confusion_matrix, writer)
