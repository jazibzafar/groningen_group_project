##
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (5,5)


##
d_tA = np.loadtxt("./output_final/direct_tA.txt")
d_tL = np.loadtxt("./output_final/direct_tL.txt")
d_vA = np.loadtxt("./output_final/direct_vA.txt")
d_vL = np.loadtxt("./output_final/direct_vL.txt")
d_test_acc = np.loadtxt("./output_final/direct_test_acc.txt")

i_tA = np.loadtxt("./output_final/india_train_acc.txt")
i_tL = np.loadtxt("./output_final/india_train_loss.txt")
i_vA = np.loadtxt("./output_final/india_val_acc.txt")
i_vL = np.loadtxt("./output_final/india_val_loss.txt")
i_test_acc = np.loadtxt("./output_final/india_test_loss.txt") # loss is actually accuracy lol

pretrain_loss = np.loadtxt("./output_final/pretrain_loss.txt")
pretrain_iou  = np.loadtxt("./output_final/pretrain_iou.txt")

##

plt.plot(d_tA, label = "COCO-pretrained Train IoU")
plt.plot(d_vA, label = "COCO-pretrained Val IoU")
plt.plot(d_test_acc, label = "COCO-pretrained Test IoU")
plt.title("Performance - COCO-pretrained")
plt.xlabel("epoch")
plt.ylabel("IoU")
plt.legend()
plt.show()

##
plt.plot(i_tA, label = "India-pretrained Train IoU")
plt.plot(i_vA, label = "India-pretrained Val IoU")
plt.plot(i_test_acc, label = "India-pretrained Test IoU")
plt.title("Performance - India-pretrained IoU")
plt.xlabel("epoch")
plt.ylabel("IoU")
plt.legend()
plt.show()

##
plt.plot(d_test_acc, label = "COCO-pretrained IoU")
plt.plot(i_test_acc, label = "India-pretrained IoU")
plt.title("Performance Comparison")
plt.xlabel("epoch")
plt.ylabel("IoU")
plt.legend()
plt.show()


##
plt.plot(i_tL, label = "India-pretrained Train loss")
plt.plot(i_vL, label = "India-pretrained Val loss")
plt.title("Loss - India-pretrained")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
plt.show()

##
plt.plot(d_tL, label = "COCO-pretrained Train loss")
plt.plot(d_vL, label = "COCO-pretrained Val loss")
plt.title("Loss - COCO-pretrained")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
plt.show()

##
plt.plot(i_tL, label = "COCO-pretrained Train loss")
plt.plot(d_tL, label = "COCO-pretrained Val loss")
plt.title("Loss Comparison")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
plt.show()

##
plt.plot(pretrain_iou, label = "India-pretraining IoU")
plt.plot(pretrain_loss, label = "India-pretraining loss")
plt.title("Pretraining Stats")
plt.xlabel("epoch")
plt.ylabel("IoU/loss")
plt.legend()
plt.show()

##
# train_loss = np.loadtxt('./output/train_loss.txt')
# val_loss = np.loadtxt('./output/val_loss.txt')
# train_iou = np.loadtxt('./output/train_iou.txt')
# val_iou = np.loadtxt('./output/val_iou.txt')

# plt.plot(train_iou, label='Train IoU')
# plt.plot(val_iou, label='Val IoU')
# plt.title("Intersection over Union")
# plt.xlabel("epoch")
# plt.ylabel("IoU")
# plt.legend()
# plt.show()
##
# plt.plot(train_loss, label='Train Loss')
# plt.plot(val_loss, label='Val Loss')
# plt.legend()
# plt.title('Loss')
# plt.xlabel('epoch')
# plt.ylabel('loss')
# plt.show()