import torch

from icecream import ic
def cal_accuracy(output, label, topk=(1,5)):
    '''
    cal accuracy for top1 and top5

    '''
    with torch.no_grad():
        # label: ([bs])
        maxk = max(topk)
        bs = label.shape[0]
        # pred_index : ([8, 5]) ([bs, maxk])
        _, pred_index = output.topk(maxk, 1, True, True)  # 按行取topk
        label_expand = label.reshape(bs, 1).expand(bs, maxk) # ([8, 5]) ([bs, maxk])
        correct = pred_index.eq(label_expand)
        result = []
        for k in topk:
            # 按照k来取出来正确的数量
            correct_k = correct[:, :k].reshape(-1).float().sum()
            result.append(correct_k.item())

        return result
            

if __name__ == "__main__":
    test_output = torch.randn(8, 100)
    test_label = torch.randint(0, 100, (8,))
    top1, top5 = cal_accuracy(test_output, test_label)
    ic(top1)
    # ic(test_label)