import torch
import torchvision

class SimLoss(torch.nn.Module):
    def __init__(self, num_classes=1, alpha=0.6):
        """
        args:
            alpha: the weight of the aug_sim_loss
        """
        super().__init__()

        self.num_classes = num_classes
        self.alpha = alpha
        self.pos_val_for_all_neg = 5
        self.neg_val_for_all_pos = -5
        self.blur = torchvision.transforms.GaussianBlur(kernel_size=9, sigma=3)
        self.sim_criterion = torch.nn.BCEWithLogitsLoss()
        
    def forward(self, input, target):
        assert input.shape == target.shape, "pred.shape must same as target.shape, but received pred.shape={0}, target.shape={1}".format(input.shape, target.shape)
        try: 
            bs, nc, height, width = input.shape
        except Exception as e:
            raise RuntimeError("expect input to have shape (bs, nc, height, width), but received {0}".format(input.shape))

        assert target.max() == 1, "target max must be 1, but received {0}".format(target.max())
        assert target.min() == 0, "target min must be 0, but received {0}".format(target.min())
        assert torch.numel(torch.unique(target)) == 2, "target should only have 0 and 1, but it had {0}".format(torch.unique(target))

        aug_sim_target_false_pos = (self.blur(target * 255) > 0).to(torch.int64)
        aug_sim_target_false_neg = (self.blur(-(target -1) * 255) == 0).to(torch.int64)

        
        input, target = input.flatten(2), target.flatten(2).float()
        aug_sim_target_false_pos = aug_sim_target_false_pos.flatten(2).float()
        aug_sim_target_false_neg = aug_sim_target_false_neg.flatten(2).float()

        
        # 如果某个feature map全阳的话，则duplicated_false_pos_map为自建值
        # 如果某个feature map全阴的话，则duplicated_pos_map为自建值
        duplicated_pos = self.get_duplicated_pos(input, target)
        duplicated_false_pos = self.get_duplicated_false_pos(input, aug_sim_target_false_pos)
        duplicated_false_neg = self.get_duplicated_false_neg(input, aug_sim_target_false_neg)

        sim = duplicated_pos * input
        aug_sim_false_pos = duplicated_pos * duplicated_false_pos
        aug_sim_false_neg = duplicated_pos * duplicated_false_neg

        sim_loss = self.sim_criterion(sim, target) + self.sim_criterion(duplicated_pos, torch.ones_like(input))
        aug_sim_loss_false_pos = self.sim_criterion(aug_sim_false_pos, torch.zeros_like(input))
        aug_sim_loss_false_neg = self.sim_criterion(aug_sim_false_neg, torch.ones_like(input))

        return sim_loss + self.alpha * aug_sim_loss_false_pos + self.alpha * aug_sim_loss_false_neg


    def get_duplicated_pos(self, input, target):
        duplicated_pos = []

        # Loop for duplicated_pos calculation
        for input_sample, target_sample in zip(input, target):
            duplicated_pos_sample = []
            
            for input_map, target_map in zip(input_sample, target_sample):
                pos_map_idx = target_map > 0
                if pos_map_idx.sum() == 0:
                    # print("all neg")
                    # 当全阴时，则不存在应当阳的情况
                    pos_map = torch.tensor([self.pos_val_for_all_neg]).to(input_map.device)
                else:
                    # 存在应当阳的情况
                    pos_map = input_map[pos_map_idx]
                
                input_over_pos_ratio = (input_map.shape[0] + pos_map.shape[0] - 1) // pos_map.shape[0]
                duplicated_pos_map = torch.tile(pos_map, [input_over_pos_ratio])[: input_map.shape[0]]
                duplicated_pos_sample.append(duplicated_pos_map.unsqueeze(0))
            
            duplicated_pos_sample = torch.cat(duplicated_pos_sample, 0)
            duplicated_pos.append(duplicated_pos_sample.unsqueeze(0))

        duplicated_pos = torch.cat(duplicated_pos, 0)
        return duplicated_pos

    def get_duplicated_false_pos(self, input, target):
        duplicated_false_pos = []

        # Loop for duplicated_false_pos calculation
        for input_sample, target_sample in zip(input, target):
            duplicated_false_pos_sample = []
            
            for input_map, target_map in zip(input_sample, target_sample):

                false_pos_map_idx = torch.logical_and(input_map > 0, torch.logical_not(target_map > 0))
                if torch.logical_not(target_map > 0).sum() == 0:
                    # print("all pos")
                    # 对于全是正例的情况是没有假阳的
                    false_pos_map = torch.tensor([self.neg_val_for_all_pos]).to(input_map.device)
                elif false_pos_map_idx.sum() == 0:
                    # print("not all pos, but not false pos")
                    # 并非全是正例，但是没有假阳的情况
                    false_pos_map = input_map[torch.logical_not(target_map > 0)]
                else:
                    # 存在假阳的情况
                    false_pos_map = input_map[false_pos_map_idx]
                
                input_over_false_pos_ratio = (input_map.shape[0] + false_pos_map.shape[0] - 1) // false_pos_map.shape[0]
                duplicated_false_pos_map = torch.tile(false_pos_map, [input_over_false_pos_ratio])[: input_map.shape[0]]
                duplicated_false_pos_sample.append(duplicated_false_pos_map.unsqueeze(0))
            
            duplicated_false_pos_sample = torch.cat(duplicated_false_pos_sample, 0)
            duplicated_false_pos.append(duplicated_false_pos_sample.unsqueeze(0))

        duplicated_false_pos = torch.cat(duplicated_false_pos, 0)
        return duplicated_false_pos
    
    def get_duplicated_false_neg(self, input, target):
        duplicated_false_neg = []

        # Loop for duplicated_false_neg calculation
        for input_sample, target_sample in zip(input, target):
            duplicated_false_neg_sample = []
            
            for input_map, target_map in zip(input_sample, target_sample):

                false_neg_map_idx = torch.logical_and(input_map < 0, target_map > 0)
                if (target_map > 0).sum() == 0:
                    # print("all neg")
                    # 对于全是负例的情况是没有假阴的
                    false_neg_map = torch.tensor([self.pos_val_for_all_neg]).to(input_map.device)
                elif false_neg_map_idx.sum() == 0:
                    # print("not all neg, but not false neg")
                    # 并非全是负例，但是没有假阴的情况
                    false_neg_map = input_map[target_map > 0]
                else:
                    # 存在假阴的情况
                    false_neg_map = input_map[false_neg_map_idx]
                
                input_over_false_neg_ratio = (input_map.shape[0] + false_neg_map.shape[0] - 1) // false_neg_map.shape[0]
                duplicated_false_neg_map = torch.tile(false_neg_map, [input_over_false_neg_ratio])[: input_map.shape[0]]
                duplicated_false_neg_sample.append(duplicated_false_neg_map.unsqueeze(0))
            
            duplicated_false_neg_sample = torch.cat(duplicated_false_neg_sample, 0)
            duplicated_false_neg.append(duplicated_false_neg_sample.unsqueeze(0))

        duplicated_false_neg = torch.cat(duplicated_false_neg, 0)
        return duplicated_false_neg