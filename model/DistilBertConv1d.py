import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import DistilBertPreTrainedModel,DistilBertModel,DistilBertConfig
from transformers.modeling_outputs import TokenClassifierOutput

class DistilBertConv1dConfig(DistilBertConfig):
    def __init__(self,conv_setting=[],**kwargs):
        super().__init__(**kwargs)
        self.conv_setting = conv_setting

class DistilBertConv1dForTokenClassification(DistilBertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.distilbert = DistilBertModel(config)
        self.dropout = nn.Dropout(config.dropout)

        self.conv1ds = nn.ModuleList(
            [nn.Conv1d(s['in_channels'],s['out_channels'],
            kernel_size=s['kernel_size'],
            padding=int((s['kernel_size']-1)/2) ) for s in config.conv_setting
            ])
        self.activation = nn.ReLU()
        self.classifier = nn.Linear(768,2)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss. Indices should be in ``[0, ..., config.num_labels -
            1]``.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.distilbert(
            input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        
        conv_output = sequence_output.transpose(1,2)
        for c in self.conv1ds[:-1]:
            conv_output = c(conv_output)
            conv_output = self.activation(conv_output)
        conv_out = self.conv1ds[-1](conv_output)
        logits = torch.transpose(conv_out,1,2).contiguous()

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
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

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

