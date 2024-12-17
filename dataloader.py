from torch.utils.data import Dataset
from collator import FinetuneDataCollatorWithPadding, EvalDataCollatorWithPadding

class RecformerTrainDataset(Dataset):
    def __init__(self, user2train, collator: FinetuneDataCollatorWithPadding):

        '''
        user2train: dict of sequence data, user--> item sequence
        '''
        
        self.user2train = user2train
        self.collator = collator
        self.users = sorted(user2train.keys())

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):

        user = self.users[index]
        seq = self.user2train[user]
        
        return seq

    def collate_fn(self, data):

        return self.collator([{'items': line} for line in data])



# class RecformerEvalDataset(Dataset):
#     def __init__(self, user2train, user2val, user2test, mode, collator: EvalDataCollatorWithPadding):
#         self.user2train = user2train
#         self.user2val = user2val
#         self.user2test = user2test
#         self.collator = collator

#         if mode == "val":
#             self.users = list(self.user2val.keys())
#         else:
#             self.users = list(self.user2test.keys())

#         self.mode = mode

#     def __len__(self):
#         return len(self.users)

#     def __getitem__(self, index):
#         user = self.users[index]
#         seq = self.user2val[user][:-1]
#         label = self.user2val[user]
#         return seq, label

#     def collate_fn(self, data):

#         return self.collator([{'items': line[0], 'label': line[1]} for line in data])
    


class RecformerEvalDataset(Dataset):
    def __init__(self, user2train, user2val, user2test, mode, collator: EvalDataCollatorWithPadding):
        self.user2train = user2train
        self.user2val = user2val
        self.user2test = user2test
        self.collator = collator

        if mode == "val":
            self.users = list(self.user2val.keys())
        else:
            self.users = list(self.user2test.keys())

        self.mode = mode

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        user = self.users[index]
        # Retrieve the sequence and label; both could have variable lengths
        seq = self.user2val[user][:-1]  # example: variable length sequence
        label = self.user2val[user]     # example: variable length label
        return seq, label

    def collate_fn(self, batch):
        # batch is a list of (seq, label) tuples
        # If collator already handles variable lengths, just pass them through:
        batch_data = [{'items': item[0], 'label': item[1]} for item in batch]
        return self.collator(batch_data)