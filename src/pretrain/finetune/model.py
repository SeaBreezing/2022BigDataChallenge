import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

from category_id_map import CATEGORY_ID_LIST


class MultiModal(nn.Module):
    def __init__(self, args, pretrain=None):
        super().__init__()
        self.bert = pretrain.roberta.bert
        self.dropout = nn.Dropout(0.3)

        
        self.classifier = nn.Linear(768, len(CATEGORY_ID_LIST))

    def forward(self, inputs, inference=False, coef=False):
        text_embedding = self.bert.embeddings(inputs['title_input'])
        cls_embedding = text_embedding[:, 0:1, :]
        cls_mask = inputs['title_mask'][:, 0:1]
        text_mask = inputs['title_mask'][:, 1:]
        text_embedding = text_embedding[:, 1:, :]
        vision_embedding = self.bert.video_fc(inputs['frame_input'])

        vision_embedding = self.bert.video_embeddings(inputs_embeds=vision_embedding)
        combine_attention_mask = torch.cat([cls_mask, inputs['frame_mask'], text_mask], dim=1)
        combine_embedding = torch.cat([cls_embedding, vision_embedding, text_embedding], dim=1)
        sequence_output = self.bert.encoder(combine_embedding, encoder_attention_mask=combine_attention_mask)[0]
        meanpooling = MeanPooling()
        final_embed = meanpooling(sequence_output,combine_attention_mask)


        final_embedding = self.dropout(final_embed)
        prediction = self.classifier(final_embedding)
        
        if inference and coef:
            return prediction
        if inference:
            return torch.argmax(prediction, dim=1)
        if coef:
            label = inputs['label']
            label = label.squeeze(dim=1)
            return prediction, label
        else:
            return self.cal_loss(prediction, inputs['label'])

    @staticmethod
    def cal_loss(prediction, label):
        label = label.squeeze(dim=1)
        loss = F.cross_entropy(prediction, label)
        with torch.no_grad():
            pred_label_id = torch.argmax(prediction, dim=1)
            accuracy = (label == pred_label_id).float().sum() / label.shape[0]
        return loss, accuracy, pred_label_id, label

class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling,self).__init__()

    def forward(self,last_hidden_state,attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded,1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask,min = 1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings
