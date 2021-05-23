import numpy as np
import matplotlib.pyplot as plt

figpath = './figures/'

def plot_loss_accuracy(loss, accuracy, modelname):
    epochs = np.arange(len(loss)) + 1
    plt.figure()
    plt.plot(loss, 'o-', color='red')
    plt.xticks(epochs)
    plt.xlabel('Training Epochs')
    plt.ylabel('Cross-Entropy Loss')
    plt.title('Loss during training')
    plt.savefig('{}{}_training_loss.png'.format(figpath, modelname))

    plt.figure()
    plt.plot(accuracy, 'o-', color='blue')
    plt.xticks(epochs)
    plt.xlabel('Training Epochs')
    plt.ylabel('Training accuracy')
    plt.title('Accuracy during training')
    plt.savefig('{}{}_training_accuracy.png'.format(figpath, modelname))

def plot_runtime_vs_minibatch(runtimes, batchsizes, optname):
    plt.figure()
    plt.style.use('ggplot')
    x_pos = [i for i, _ in enumerate(batchsizes)]
    plt.bar(x_pos, runtimes, color='green')
    plt.xlabel('Batch size')
    plt.ylabel('Convergence runtime (secs)')
    plt.title('Runtime vs batchsize')
    plt.xticks(x_pos, batchsizes)
    plt.savefig('{}{}_runtime_vs_minibatch.png'.format(figpath, optname))
    