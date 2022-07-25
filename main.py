import argparse
from data_interface import DataInterface
from model import resnet50
import torch.optim as optim
import torch.nn as nn
import torch
import torchvision

def train(args: argparse.Namespace) ->None:
    mydata = DataInterface(**vars(args))

    train_loader = mydata.train_dataloader()
    test_loader = mydata.test_dataloader()
    classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    lr = args.lr
    epochs = 3
    device = args.device
    best_acc = 0.0
    save_path = './resNet50.pth'

    net = resnet50(classes=10).to(device)
    # 定义优化器和损失函数
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=lr)
    loss_function = nn.CrossEntropyLoss()
    
    
    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0
        best_loss = [0,0,11111]
        for step, data in enumerate(train_loader):
            images, labels = data

            optimizer.zero_grad()
            outputs  = net(images.to(device))
            loss = loss_function(outputs, labels.to(device))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            print(f'[{epoch + 1}, {step + 1:5d}] loss: {loss :.3f}')

            if step % 2000 == 1999:    # print every 2000 mini-batches
                if best_loss[2] > running_loss:
                    best_loss = [epoch+1,step + 1,running_loss]
                    torch.save(net.state_dict(), './resNet50_best.pth')
                print("********************************************************************************************************")
                print(f'[{epoch + 1}, {step + 1:5d}] loss: {running_loss / 2000:.3f}')
                print("********************************************************************************************************")
                running_loss = 0.0
                torch.save(net.state_dict(), save_path)
        print(best_loss)
    print('Finished Training')
        # validate
        # net.eval()
        # acc = 0.0  # accumulate accurate number / epoch
        # with torch.no_grad():
        #     for val_data in test_loader:
        #         val_images, val_labels = val_data
        #         outputs = net(val_images.to(device))
        #         # loss = loss_function(outputs, test_labels)
        #         predict_y = torch.max(outputs, dim=1)[1]
        #         acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

        #         print("valid epoch[{}/{}]".format(epoch + 1,epochs))
        # val_accurate = acc / val_num
        #         print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
        #             (epoch + 1, running_loss / train_steps, val_accurate))

def test(args: argparse.Namespace) ->None:
    pass

def parse_opt() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--device', type=str, default='cuda:2')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.0001)


    opt = parser.parse_args()
    return opt

if __name__ == '__main__':
    opt = parse_opt()
    if opt.mode == 'train':
        train(opt)
    else:
        test(opt)

   
