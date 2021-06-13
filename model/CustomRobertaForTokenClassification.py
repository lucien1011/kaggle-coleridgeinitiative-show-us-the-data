import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import BertPreTrainedModel,RobertaModel,RobertaConfig
from transformers.modeling_outputs import TokenClassifierOutput

class CustomRobertaForTokenClassification(BertPreTrainedModel):
    config_class = RobertaConfig
    base_model_prefix = "roberta"

    def __init__(self, config , class_weight=None, use_all_attention=True):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.class_weight = class_weight
        self.use_all_attention = use_all_attention
        if self.use_all_attention:
            self.classifier = nn.Linear(config.num_hidden_layers*config.hidden_size, config.num_labels)
        else:
            self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.dummy_param = nn.Parameter(torch.empty(0))

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        sample_weight=None,
    ):

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
 
        if self.use_all_attention:
            sequence_output = torch.cat(outputs.hidden_states[1:],axis=2)
        else:
            sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            reduction = 'mean' if sample_weight is None else 'none'
            class_weight = torch.tensor(self.class_weight).to(self.dummy_param.device)
            loss_fct = CrossEntropyLoss(weight=class_weight)
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            if sample_weight is not None:
                loss = loss * sample_weight
                loss = loss.mean()

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

