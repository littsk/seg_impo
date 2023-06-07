import torch

class SimLoss(torch.nn.Module):
    def __init__(self, num_classes=1, alpha=0.5):
        """
        args:
            alpha: the weight of the sim_loss, (1 - alpha) is the weight of ce loss
        """
        super().__init__()

        self.num_classes = num_classes
        self.alpha = alpha

        self.sim_criterion = torch.nn.BCEWithLogitsLoss()

        self.ce_criterion = torch.nn.CrossEntropyLoss() if self.num_classes > 1 else torch.nn.BCEWithLogitsLoss()
        
    def forward(self, input, target):
        assert input.shape == target.shape, "pred.shape must same as target.shape"
        try: 
            bs, nc, height, width = input.shape
        except Exception as e:
            raise ValueError("expect input to have shape (bs, nc, height, width), but received {0}".format(input.shape))
        
        # dice loss
        # if(nc > 1): 
        #     # 多分类的情况
        #     dice_loss = 1 - self.dice_coeff(torch.softmax(input, dim=1), target)
        # else:
        #     dice_loss = 1 - self.dice_coeff(torch.sigmoid(input), target)
        ce_loss = self.ce_criterion(input, target.float())
        sim_loss = self.sim_loss(input, target)
        return self.alpha * sim_loss + (1 - self.alpha) * ce_loss
    
    def sim_loss(self, input, target):
        # 遍历batch中的每一个样本
        input, target = input.flatten(2), target.flatten(2).float()

        duplicated_pos = []
        for input_sample, target_sample in zip(input, target):
            
            # 在多类的分割任务中，还需要遍历每一个类的feature map
            duplicated_pos_sample = []
            for input_map, target_map in zip(input_sample, target_sample):
                pos_map = input_map[target_map > 0]

                input_over_pos_ratio = (input_map.shape[0] + pos_map.shape[0] - 1) // pos_map.shape[0]
                duplicated_pos_map = torch.tile(pos_map, [input_over_pos_ratio])[: input_map.shape[0]]

                duplicated_pos_sample.append(duplicated_pos_map.unsqueeze(0))
            duplicated_pos_sample = torch.concat(duplicated_pos_sample, 0)
            duplicated_pos.append(duplicated_pos_sample.unsqueeze(0))
        duplicated_pos = torch.concat(duplicated_pos, 0)

        # 计算相似度
        output = duplicated_pos * input

        # distance = torch.sqrt(torch.square(duplicated_pos - input).mean())
        return self.sim_criterion(output, target) + self.sim_criterion(duplicated_pos, torch.ones_like(duplicated_pos))

    
    def dice_coeff(self, input, target, reduce_batch_first=False, epsilon=1e-6):
        # Average of Dice coefficient for all batches, or for a single mask
        input, target = input.flatten(0, 1), target.flatten(0, 1)

        sum_dim = (-1, -2) if not reduce_batch_first else (-1, -2, -3)

        inter = 2 * (input * target).sum(dim=sum_dim)
        sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
        sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

        dice = (inter + epsilon) / (sets_sum + epsilon)
        return dice.mean()