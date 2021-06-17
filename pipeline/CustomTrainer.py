from nlptools.Trainer import Trainer

class CustomTrainer(Trainer):
    @classmethod
    def patch_train_batch(cls,batch):
        return {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "sample_weight": batch[3],
                "labels": batch[-1],
                }

    @classmethod
    def patch_val_batch(cls,batch):
        return {"input_ids": batch[0],"attention_mask": batch[1],"labels": batch[2]}

