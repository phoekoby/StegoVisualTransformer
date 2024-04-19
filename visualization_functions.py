import io

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.utils as utils
from PIL import Image
from torchvision.transforms import ToTensor



def write_to_tensor(name, buf, writer):
    image = Image.open(buf)
    image = ToTensor()(image).unsqueeze(0)
    writer.add_image(name, image, dataformats='NCHW')


def Plt_hist(loss_history, train_history, val_history, val_losses, writer):
    plt.plot(train_history)
    plt.plot(val_history)
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.grid()

    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg')
    buf.seek(0)

    plt.show()

    write_to_tensor("Accuracy plt", buf, writer)

    plt.plot(loss_history)
    plt.plot(val_losses)
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['model loss', 'val loss'], loc='upper left')
    plt.grid()

    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg')
    buf.seek(0)

    plt.show()

    write_to_tensor("Loss plt", buf, writer)


def visualize(model, test_loader, writer, device, batch_size, algorithm):
    # model.load_state_dict(torch.load(args.checkpoint))
    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(test_loader):
            # get images
            inputs = inputs.to(device)
            # if batch_idx == 0:
            images = inputs[0:batch_size, :, :, :]
            I = utils.make_grid(images, nrow=8)
            writer.add_image('origin', I)
            # _, c_crm, c1, c2, c3 = model(images)
            _, c1, c2, c3 = model(images)
            # print(I.shape, c1.shape, c2.shape, c3.shape, c_crm.shape)
            # attn_crm = visualize_attn(I, c_crm)
            # writer.add_image('attn_crm', attn_crm)
            attn1 = visualize_attn(I, c1)
            writer.add_image(f'1 Слой внимания {algorithm}', attn1)
            attn2 = visualize_attn(I, c2)
            writer.add_image(f'2 Слой внимания {algorithm}', attn2)
            attn3 = visualize_attn(I, c3)
            writer.add_image(f'3 Слой внимания {algorithm}', attn3)
            break
    print('Visualization complete!')
    writer.close()


def build_confusion_matrix(predictions, ground_truth):
    confusion_matrix = np.zeros((2, 2), np.int_)
    for i in range(len(predictions)):
        confusion_matrix[predictions[i].cpu()][ground_truth[i]] += 1
    return confusion_matrix


def visualize_confusion_matrix(confusion_matrix, writer):
    size = confusion_matrix.shape[0]
    fig = plt.figure(figsize=(2, 2))
    plt.title("Confusion matrix")
    plt.ylabel("predicted")
    plt.xlabel("ground truth")
    res = plt.imshow(confusion_matrix, cmap='GnBu', interpolation='nearest')
    cb = fig.colorbar(res)
    plt.xticks(np.arange(size))
    plt.yticks(np.arange(size))
    for i, row in enumerate(confusion_matrix):
        for j, count in enumerate(row):
            plt.text(j, i, count, fontsize=14, horizontalalignment='center', verticalalignment='center')

    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg')
    buf.seek(0)

    plt.show()
    write_to_tensor("confusion_matrix", buf, writer)
