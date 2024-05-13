"""Train the label-dependent or label-independent character representation models"""

import os
import pandas as pd
from typing import Any

class Trainer:

    def __init__(self,
                 data_dir,
                 splits_file,
                 data_type,
                 pretrained_model_name,
                 tokenizer_model_name,
                 ) -> None:
        self.data_dir = data_dir
        self.splits_file = splits_file
        self.data_type = data_type
        self.pretrained_model_name = pretrained_model_name
        self.tokenizer_model_name = tokenizer_model_name

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        splits_file = os.path.join(self.data_dir, self.splits_file)
        splits_df = pd.read_csv(splits_file, index_col=None, dtype={"imdb-id": str, "content-text": str})
        tensors_dir = os.path.join(self.data_dir, "30-modeling/tensors", self.data_type, self.tokenizer_model_name)

        # encode traits