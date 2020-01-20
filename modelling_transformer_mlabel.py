import transformers
from torch import nn
import torch
import sys

class MlabelSimple(nn.Module):

    def __init__(self,bert,label_count):
        super().__init__()
        self.encoder=bert
        self.classifier=nn.Linear(self.encoder.config.hidden_size,label_count).cuda()

    def forward(self,encoder_input):
        last_hidden,cls=self.encoder(encoder_input)
        return torch.sigmoid(self.classifier(cls))
        #classification_output=self.classifier(bert_encoded
        
        
class MlabelDecoder(nn.Module):

    @classmethod
    def from_bert(cls,bert,decoder_config_file):
        decoder=transformers.modeling_bert.BertModel(transformers.configuration_bert.BertConfig.from_json_file(decoder_config_file))
        decoder=decoder.cuda()
        return cls(bert,decoder)
    
    def __init__(self,encoder,decoder):
        super().__init__()
        self.encoder=encoder
        self.decoder=decoder
        self.classifier=nn.Linear(self.decoder.config.hidden_size,1)
        

    def forward(self,inp_classes,encoder_input=None,encoder_output=None,encoder_attention_mask=None):
        """
        'inp_classes': input to the decoder part
        'encoder_input': the input data point itself
        'encoder_output': the output of the encoder (if None, it will be calculated for you)
        'encoder_attention_mask': If None, it will be all-ones, ie whole encoded input can be seen
        """
        if encoder_output is None:
            if encoder_attention_mask is None:
                encoder_attention_mask=torch.ones_like(encoder_input)
            #print(encoder_input)
            #print(encoder_input.shape)
            preds_t=self.encoder(encoder_input,attention_mask=encoder_attention_mask)
            encoder_output=preds_t[2][-1] #TODO! VERIFY! preds_t[2] is the hidden_state_output, and [-1] is the last encoder layer
        positions=torch.ones_like(inp_classes)
        decoder_output=self.decoder(inp_classes,attention_mask=torch.ones_like(inp_classes),encoder_hidden_states=encoder_output,encoder_attention_mask=encoder_attention_mask,position_ids=positions)
        return encoder_output,encoder_attention_mask,decoder_output
    

    
