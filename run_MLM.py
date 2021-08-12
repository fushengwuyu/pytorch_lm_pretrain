# author: sunshine
# datetime:2021/8/11 下午2:43
import torch
from transformers import AutoModelWithLMHead, AutoTokenizer, LineByLineTextDataset, TextDataset, \
    DataCollatorForLanguageModeling, PreTrainedModel
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers.trainer_pt_utils import get_parameter_names
from argparse import Namespace
from torch.utils.data import DataLoader
import math
import logging
import os

logger = logging.getLogger(__name__)


def get_args():
    params = dict(
        eval_data_file='eval.txt',
        train_data_file='train.txt',
        epoch_num=100,
        batch_size=4,
        drop_last=False,
        line_by_line=True,
        bert_path='/home/sunshine/pre_models/pytorch/ernie-1.0',
        output='output',
        mlm_probability=0.15,
        dataloader_num_workers=0,
        dataloader_pin_memory=True,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-08,
        warmup_ratio=0,
        warmup_steps=0,
        weight_decay=0.0
    )
    return Namespace(**params)


def get_dataset(args, tokenizer, evaluate=False):
    file_path = args.eval_data_file if evaluate else args.train_data_file
    if args.line_by_line:
        return LineByLineTextDataset(tokenizer=tokenizer, file_path=file_path, block_size=args.batch_size)
    else:
        return TextDataset(
            tokenizer=tokenizer, file_path=file_path, block_size=args.batch_size, overwrite_cache=args.overwrite_cache
        )


def create_dataset(args, tokenizer):
    train_dataset = get_dataset(args, tokenizer=tokenizer)
    eval_dataset = get_dataset(args, tokenizer=tokenizer, evaluate=True)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=args.mlm_probability
    )
    train_data_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        collate_fn=data_collator,
        num_workers=args.dataloader_num_workers,
        pin_memory=args.dataloader_pin_memory,
    )

    dev_data_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        collate_fn=data_collator,
        num_workers=args.dataloader_num_workers,
        pin_memory=args.dataloader_pin_memory,
    )

    return [train_data_loader, dev_data_loader]


class Trainer(object):

    def __init__(self, args, data_loader, tokenizer=None):

        self.args = args
        self.tokenizer = tokenizer
        self.device = torch.device("cuda:{}".format(args.device_id) if torch.cuda.is_available() else "cpu")
        self.model = AutoModelWithLMHead.from_pretrained(args.bert_path)
        self.model.to(self.device)

        self.train_data_loader, self.dev_data_loader = data_loader
        self.optimizer, self.schedule = self.set_optimizer(
            num_training_steps=(
                    len(self.train_data_loader) * args.epoch_num))

    def set_optimizer(self, num_training_steps=None):

        decay_parameters = get_parameter_names(self.model, [torch.nn.LayerNorm])
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if n in decay_parameters],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if n not in decay_parameters],
                "weight_decay": 0.0,
            },
        ]

        optimizer_kwargs = {
            "betas": (self.args.adam_beta1, self.args.adam_beta2),
            "eps": self.args.adam_epsilon,
        }
        optimizer = AdamW(optimizer_grouped_parameters, **optimizer_kwargs)

        warmup_steps = (
            self.args.warmup_steps
            if self.args.warmup_steps > 0
            else math.ceil(num_training_steps * self.args.warmup_ratio)
        )

        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
        )

        return optimizer, lr_scheduler

    def train(self):

        self.model.train()
        step_gap, eval_gap = 20, 250
        for epoch in range(int(self.args.epoch_num)):

            best_loss, gap_loss = 0.0, 1e10

            for step, batch in enumerate(self.train_data_loader):

                batch = {k: v.to(self.device) for k, v in batch.items()}

                loss = self.train_step(batch)
                gap_loss += loss
                if step % step_gap == 0:
                    current_loss = gap_loss / step_gap
                    msg = u"step {} / {} of epoch {}, train/loss: {}".format(step, len(self.train_data_loader),
                                                                             epoch, current_loss)
                    print(msg)

            if current_loss < best_loss:
                self.save(output_dir="{}/{}".format(self.args.outpu, epoch))

    def unwrap_model(self, model: torch.nn.Module) -> torch.nn.Module:
        """
        Recursively unwraps a model from potential containers (as used in distributed training).

        Args:
            model (:obj:`torch.nn.Module`): The model to unwrap.
        """
        # since there could be multiple levels of wrapping, unwrap recursively
        if hasattr(model, "module"):
            return self.unwrap_model(model.module)
        else:
            return model

    def save(self, output_dir=None, state_dict=None, weight_name="pytorch_model.bin"):
        """模型保存

        """
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not isinstance(self.model, PreTrainedModel):
            if isinstance(self.unwrap_model(self.model), PreTrainedModel):
                if state_dict is None:
                    state_dict = self.model.state_dict()
                self.unwrap_model(self.model).save_pretrained(output_dir, state_dict=state_dict)
            else:
                logger.info("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
                if state_dict is None:
                    state_dict = self.model.state_dict()
                torch.save(state_dict, os.path.join(output_dir, weight_name))
        else:
            self.model.save_pretrained(output_dir, state_dict=state_dict)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

    def train_step(self, inputs):

        output = self.model(**inputs)
        loss = output.loss
        loss.backward()
        loss = loss.item()
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.schedule.step()
        self.model.zero_grad()
        return loss


def main():
    args = get_args()

    tokenizer = AutoTokenizer.from_pretrained(args.bert_path)
    data_loader = create_dataset(args, tokenizer)

    trainer = Trainer(
        args, data_loader
    )

    trainer.train()


if __name__ == '__main__':
    main()
