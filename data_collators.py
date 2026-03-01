from dataclasses import dataclass
from typing import Any, Dict, List, Union

import torch

@dataclass
class DataCollatorSpeechSeq2Seq:
    processor: Any
    decoder_start_token_id: int
 
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{'input_features': feature['input_features']} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors='pt')

        if 'attention_mask' in features[0]:
            batch['attention_mask'] = torch.stack([torch.tensor(f['attention_mask']) for f in features])
 
        label_features = [{'input_ids': feature['labels']} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors='pt')
 
        labels = labels_batch['input_ids'].masked_fill(labels_batch.attention_mask.ne(1), -100)
 
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]
 
        batch['labels'] = labels
 
        return batch
