import gc

import torch
from torch import nn
from transformers import DistilBertPreTrainedModel,DistilBertModel
from transformers.modeling_outputs import TokenClassifierOutput

class DistilBertCNNForTokenClassification(DistilBertPreTrainedModel):

    def __init__(self,config):

        super().__init__(config)
        self.num_labels = config.num_labels

        self.distilbert = DistilBertModel(config)
        self.dropout = nn.Dropout(config.dropout)

        self.conv = nn.Conv2d(
                #in_channels=config.max_position_embeddings, out_channels=config.max_position_embeddings, 
                in_channels=1, out_channels=1, 
                kernel_size=config.conv_kernel_size, 
                padding=True,
                )
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=config.pool_kernel_size)
        self.fc = nn.Linear(
                #int((config.dim-config.conv_kernel_size)/config.pool_kernel_size), 
                config.linear_size,
                config.num_labels*config.max_position_embeddings,
                )
        self.flat = nn.Flatten(1,3)
        self.num_labels = config.num_labels

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
        sequence_output = torch.unsqueeze(sequence_output,axis=1)

        #sequence_output = torch.transpose(sequence_output,1,2)
        classifier_out = self.conv(sequence_output)
        classifier_out = self.relu(classifier_out)
        classifier_out = self.dropout(classifier_out)
        classifier_out = self.pool(classifier_out)
        classifier_out = self.dropout(classifier_out)
        #classifier_out = torch.squeeze(classifer_out)
        classifier_out = self.flat(classifier_out)
        logits = self.fc(classifier_out)
        logits = torch.reshape(logits,(len(logits),self.config.max_position_embeddings,self.num_labels))

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
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
