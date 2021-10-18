import torch
import torch.nn.functional as F

from transformers import BertTokenizer
from torch.utils.data import TensorDataset
#from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, SequentialSampler

from transformers import BertForSequenceClassification

import numpy as np

class Evaluate:

    def __init__(self):

        self.LABEL = {  'Animals': 4,
                        'Compliment': 7,
                        'Education': 10,
                        'Health': 3,
                        'Heavy Emotion': 2,
                        'Joke': 6,
                        'Love': 1,
                        'Politics': 0,
                        'Religion': 8,
                        'Science': 5,
                        'Self': 9}

        self.model = BertForSequenceClassification.from_pretrained("bert-base-uncased",
                                                      num_labels=len(self.LABEL),
                                                      output_attentions=False,
                                                      output_hidden_states=False)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(device)

        self.model.load_state_dict(torch.load('./prediction/model/finetuned_BERT_epoch_1.model', map_location=torch.device('cpu')))


    def encoding_data(self,text):

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', 
                                          do_lower_case=True)

        encoded_data = tokenizer.batch_encode_plus(
            text, 
            add_special_tokens=True, 
            return_attention_mask=True, 
            pad_to_max_length=True, 
            max_length=256, 
            return_tensors='pt'
        )


        input_ids = encoded_data['input_ids']
        attention_masks = encoded_data['attention_mask']

        return input_ids, attention_masks
    
    def data_loader(self,input_ids, attention_masks, batch_size=3):

        test_dataset = TensorDataset(input_ids, attention_masks)
        test_sampler = SequentialSampler(test_dataset)
        test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=batch_size)

        return test_dataloader
    
    def evaluate(self, test_dataloader):
    
        self.model.eval()
        
        
        predictions = []
        
        for batch in test_dataloader:
            inputs = {'input_ids':      batch[0],
                    'attention_mask': batch[1],
                    }

            with torch.no_grad():        
                outputs = self.model(**inputs)
    
            logits = outputs[0]
            logits = logits.detach().cpu().numpy()
            predictions.append(logits)
        
        predictions = np.concatenate(predictions, axis=0)
                
        return predictions

    def predict(self,text):

        input_ids, attention_masks = self.encoding_data(text)
        print("---------------input_ids completed---------------------- ")

        test_dataloader = self.data_loader(input_ids, attention_masks)
        print("---------------test_dataloader completed---------------------- ")

        output = self.evaluate(test_dataloader)
        print("---------------evaluate completed---------------------- ")
        prediction = np.argmax(output, axis=1).flatten()[0]

        print("----------------------output---------------------------------")
        print(prediction)
        topic = list(self.LABEL.keys())[list(self.LABEL.values()).index(prediction)]
        print(topic)
        print("----------------------topic printed---------------------------------")

        return topic
        


    


