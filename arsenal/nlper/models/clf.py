import torch
from transformers import (
    AutoTokenizer,
    get_scheduler,
    default_data_collator,
    DataCollatorWithPadding
)
from datasets import load_dataset
from torch.utils.data import DataLoader
from arsenal.nlper.mpl import MplModule, MplOutput


class MplCLF(MplModule):
    def training_step(self, batch):
        outputs = self.model(**batch)
        return MplOutput(loss=outputs.loss)

    def validation_step(self, batch, accelerator):
        outputs = self.model(**batch)
        predictions = outputs.logits.argmax(dim=-1)
        predictions, references = accelerator.gather_for_metrics((predictions, batch['labels']))
        return MplOutput(
            loss = outputs.loss.detach().float(),
            predictions = predictions,
            references = references
        )

    def test_step(self, batch, accelerator):
        outputs = self.model(**batch)
        predictions = outputs.logits.argmax(dim=-1)
        predictions = accelerator.gather_for_metrics(predictions)
        references = None
        loss = None
        if 'labels' in batch:
            references = accelerator.gather_for_metrics(batch['labels'])
            loss = outputs.loss.detach().float()
        return MplOutput(
            loss = loss,
            predictions = predictions,
            references = references
        )

    def prepare_data(self, use_fp16=False):
        if self.config.pad_to_max_length:
            self.data_collator = default_data_collator
        else:
            self.data_collator = DataCollatorWithPadding(self.tokenizer, pad_to_multiple_of=(8 if use_fp16 else None))

    def preprocess_function(self, examples):
        # Tokenize the texts
        config = self.config
        padding = "max_length" if config.pad_to_max_length else False
        texts = (
            (examples[config.sentence1_key],)
            if config.sentence2_key is None else (examples[config.sentence1_key], examples[config.sentence2_key])
        )
        result = self.tokenizer(*texts, padding=padding, max_length=config.max_length, truncation=True)

        if config.label_key in examples:
            if self.label_to_id is not None:
                # Map labels to IDs (not necessary for GLUE tasks)
                result["labels"] = [self.label_to_id[l] for l in examples[config.label_key]]
            else:
                # In all cases, rename the column to labels because the model will expect that.
                result["labels"] = examples[config.label_key]
        return result

    def get_train_data(self, train_file, label_key, batch_size, cache_dir, *args, **kwargs) -> MplOutput:
        """加载本地json/csv数据，将所有label映射到数字"""
        config = self.config
        label_list = None

        extension = train_file.split(".")[-1]
        dataset = load_dataset(extension, data_files={'train': train_file}, cache_dir=cache_dir)
        train_dataset = dataset['train']

        is_regression = train_dataset.features[label_key].dtype in ['float32', 'float64']
        if not is_regression:
            label_list = train_dataset.unique(label_key)
            label_list.sort()
            if not config.num_labels or config.num_labels != len(label_list):
                config.num_labels = len(label_list)
        # 将所有label映射到数字，preprocess_function中会用到
        self.label_to_id = None
        if label_list is not None:
            self.label_to_id = {v: i for i, v in enumerate(label_list)}

        train_dataset = train_dataset.map(
            self.preprocess_function,
            batched=True,
            remove_columns=train_dataset.column_names,
            desc="Run tokenizer on train dataset"
        )
        train_dataloader = DataLoader(
            train_dataset, shuffle=True, collate_fn=self.data_collator, batch_size=batch_size
        )
        return MplOutput(
            dataloader = train_dataloader,
            dataset = train_dataset
        )

    def get_eval_data(self, eval_file, batch_size, cache_dir, *args, **kwargs) -> MplOutput:
        extension = eval_file.split(".")[-1]
        dataset = load_dataset(extension, data_files={'eval': eval_file}, cache_dir=cache_dir)
        eval_dataset = dataset['eval']

        eval_dataset = eval_dataset.map(
            self.preprocess_function,
            batched=True,
            remove_columns=eval_dataset.column_names,
            desc="Run tokenizer on eval dataset"
        )

        eval_dataloader = DataLoader(
            eval_dataset, collate_fn=self.data_collator, batch_size=batch_size
        )
        return MplOutput(
            dataloader = eval_dataloader,
            dataset = eval_dataset
        )

    def get_test_data(self, test_file, label_key, batch_size, cache_dir, *args, **kwargs) -> MplOutput:
        """考虑到可能会被独立调用，所以有些参数需要重新计算"""
        config = self.config
        label_list = None

        extension = test_file.split(".")[-1]
        dataset = load_dataset(extension, data_files={'test': test_file}, cache_dir=cache_dir)
        test_dataset = dataset['test']

        # 标识测试数据是否含标签，便于之后将预测结果转换为原标签
        test_has_label = label_key in test_dataset.features if label_key else False
        if test_has_label:
            is_regression = test_dataset.features[label_key].dtype in ["float32", "float64"]
        else:
            if config.num_labels is None:
                raise ValueError("no label in test_data, you must set `--num_labels`")
            is_regression = config.num_labels == 1

        if not is_regression and test_has_label:
            # 测试集存在分类标签的情况下，检查--num_labels和分类标签数是否相同
            label_list = test_dataset.unique(label_key)
            label_list.sort()
            if config.num_labels and config.num_labels != len(label_list):
                raise ValueError(
                    f"num label of test dataset is not matched `--num_labels`, "
                    f"{len(label_list)} != {config.num_labels}, "
                    f"please check manually, test dataset label list: {label_list}"
                )
        # 将所有label映射到数字，preprocess_function中会用到
        self.label_to_id = None
        if label_list is not None:
            self.label_to_id = {v: i for i, v in enumerate(label_list)}

        test_dataset = test_dataset.map(
            self.preprocess_function,
            batched=True,
            remove_columns=test_dataset.column_names,
            desc="Run tokenizer on test dataset"
        )

        test_dataloader = DataLoader(
            test_dataset, collate_fn=self.data_collator, batch_size=batch_size
        )
        return MplOutput(
            dataloader = test_dataloader,
            dataset = test_dataset
        )
