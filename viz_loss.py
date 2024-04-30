##
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (5,5)

##
train_loss = np.loadtxt('./output/train_loss.txt')
val_loss = np.loadtxt('./output/val_loss.txt')
train_iou = np.loadtxt('./output/train_iou.txt')
val_iou = np.loadtxt('./output/val_iou.txt')
##
# plt.plot(train_iou, label='Train IoU')
# plt.plot(val_iou, label='Val IoU')
# plt.title("Intersection over Union")
# plt.xlabel("epoch")
# plt.ylabel("IoU")
# plt.legend()
# plt.show()
##
plt.plot(train_loss, label='Train Loss')
plt.plot(val_loss, label='Val Loss')
plt.legend()
plt.title('Loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()