"""
This file defines the data transforms that will be applied to the data. 
Each transform takes in an EasyDict object of in_features (key: feature_name, value: feature data)
It should output an EasyDict object of out_features (key: feature_name, value: feature_data)
Each transform defined here can be used as an independent unit to form a data pipeline
Some common transforms are provided by runway
"""
import sys
sys.path.append('../..')

from runway_for_ml.data_module.data_transforms import BaseTransform, HFDatasetTransform, register_transform_functor
from datasets import load_dataset
from torchvision.transforms import Compose, ColorJitter, ToTensor
from transformers import AutoTokenizer


# NLP Example
@register_transform_functor
class LoadMRPCDataset(BaseTransform):
    """Loads the Microsoft Research Paraphrase Corpus (MRPC) dataset https://huggingface.co/datasets/glue/viewer/mrpc 
    """
    def setup(self, *args, **kwargs):
        self.dataset = load_dataset("glue", "mrpc")
        
    def _call(self, *args, **kwargs):
        return self.dataset

@register_transform_functor
class TokenizeMRPC(HFDatasetTransform):
    """Tokenize the MRPC dataset for model training
    """
    def setup(self, tokenizer_name, *args, **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    def encode(examples):
        return self.tokenizer(examples["sentence1"], examples["sentence2"], truncation=True, padding="max_length")
    
    def _call(self, dataset, *args, **kwargs):
        """The actual transformation

        Args:
            dataset (HuggingFace DatasetDict): DatasetDict taken from the previous node

        Returns:
            DatasetDict: Tokenized datasets
        """
        for split in ['train', 'test', 'validation']:
            dataset[split] = dataset[split].map(self.encode) # apply encode function
            dataset = dataset.map(lambda examples: {"labels": examples["label"]}, batched=True) # rename to labels
        return dataset 


# CV example
@register_transform_functor
class LoadBeansDataset(BaseTransform):
    def setup(self, *args, **kwargs):
        if self.use_dummy_data:
            train_dataset = load_dataset("beans", split="train[:10]")
            test_dataset = load_dataset("beans", split="test[:10]")
            valid_dataset = load_dataset("beans", split="validation[:10]")
            self.dataset = {
                'train': train_dataset,
                'test': test_dataset,
                'validation': valid_dataset,
            }
        else:
            self.dataset = load_dataset("beans")
    
    def _call(self, input_data):
        return self.dataset

@register_transform_functor
class BeansJitterTransform(HFDatasetTransform):
    """Transform Functor to apply Jitter Transformation on Beans dataset

    """
    def setup(self, brightness=0.5, hue=0.5, *args, **kwargs):
        """Setup the Transform Functor

        Args:
            brightness (float, optional): argument for the Jitter transform. Defaults to 0.5.
            hue (float, optional): argument for the Jitter transform. Defaults to 0.5.
        """
        self.brightness = brightness
        self.hue = hue
        self.jitter = Compose([ColorJitter(brightness=0.5, hue=0.5), ToTensor()])

    
    def _call(self, input_ds):
        """The actual transform

        Args:
            input_ds (HuggingFace DatasetDict object): contains the actual dataset
        
        Returns:
            DatasetDict: processed datasets
        """
        train_dataset = input_ds['train']
        def _transform_image(examples):
            examples["pixel_values"] = [self.jitter(image.convert("RGB")) for image in examples["image"]]
            return examples
        input_ds['train'] = train_dataset.with_transform(_transform_image)
        train_dataset = train_dataset.remove_columns("image")
        # input_ds['train'] = [exp for exp in input_ds['train'][:100]]
        return train_dataset

