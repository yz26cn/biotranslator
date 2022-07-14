from dataclasses import dataclass
import transformers
from transformers import (
HfArgumentsParser,
ModelArguments,
DataTrainingArguments,
TrainingArguments,

)

class TrainingArguments(TrainingArguments):
    """
    Arguments to specify the which model we are going to train from scratch or finetune for each task.
    """
