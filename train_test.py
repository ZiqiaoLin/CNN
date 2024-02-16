from cnn import cal_correction


def train_model(model, train_data, val_data, batch, epoch, loss_func, opt):
    """

    :param model:  需要训练的模型
    :param train_data: 训练数据
    :param val_data: 验证数据
    :param batch: 每次训练的批量
    :param epoch: 训练轮数
    :param loss_func: 损失函数
    :param opt: 优化器
    :return:
    """

    for i in range(epoch):

        train_corrections = []
        train_losses = []

        for idx, (img, label) in enumerate(train_data):

            # 换转成cuda 类型
            img, label = img.to('cuda'), label.to('cuda')

            img = img.clone().requires_grad_(True)
            label = label.clone().detach()

            """ 前向传播"""

            model.train()
            output = model(img)

            train_acc = cal_correction(output, label)
            train_corrections.append(train_acc)

            train_loss = loss_func(output, label)
            train_losses.append(train_loss)

            # 晴空优化器提督
            opt.zero_grad()

            """反向传播"""
            train_loss.backward()
            # 用优化器梯度下降：
            opt.step()

            if idx % (batch * 100) == 0:
                model.eval()
                val_record = []
                for (data, target) in val_data:
                    # 转化数据类型
                    data, target = data.to('cuda'), target.to('cuda')
                    data, target = data.clone().requires_grad_(True), target.clone().detach()

                    # 将数据喂入：
                    out = model(data)
                    # 预测准确率
                    val_acc = cal_correction(out, target)
                    val_record.append(val_acc)

                    # 打印训练、验证结果
                    print(f'epoch{i + 1}:Train Acc = {train_corrections[-1]} Train Loss = {train_losses[-1]} ')


def test_model(model, test_data):
    """ 测试模型
    :param model:
    :param test_data:
    :return:
    """
    test_acc = []
    for idx, (img, label) in enumerate(test_data):
        # 处理数据
        img, label = img.to('cuda'), label.to('cuda')
        img, label = img.clone().requires_grad_(True), label.clone().detach()

        output = model(img)

        acc = cal_correction(output, label)
        test_acc.append(acc)
        print(f'Test Acc = {test_acc[-1]}%')

