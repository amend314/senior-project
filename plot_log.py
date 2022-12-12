import csv
import matplotlib.pyplot as plt


log = 'data/logs/lstm-training-1670462804.8919058.log'
with open(log) as x:
    reader = csv.reader(x)
    next(reader, None)
    accuracies = []
    val_accuracies = []
    losses = []
    val_losses = []

    for epoch, acc, loss, val_acc, val_loss in reader:
        accuracies.append(float(acc))
        val_accuracies.append(float(val_acc))
        losses.append(float(loss))
        val_losses.append(float(val_loss))

    plt.plot(accuracies, label='Accuracy')
    plt.plot(val_accuracies, label='Val_accuracy')
    plt.legend(loc='upper left')
    plt.title('Accuracies')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.show()

    plt.plot(losses, label='Loss')
    plt.plot(val_losses, label='Val_loss')
    plt.legend(loc='upper left')
    plt.title('Losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()
