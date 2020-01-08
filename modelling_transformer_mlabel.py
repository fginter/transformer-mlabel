import transformers
from torch import nn

class MlabelDecoder(nn.Module):

    @classmethod
    def from_bert(cls,bert,decoder_config_file):
        decoder=transformers.modeling_bert.BertForTokenClassification(decoder_config_file)
        return cls(bert,decoder)
    
    def __init__(self,encoder,decoder):
        self.encoder=encoder
        self.decoder=decoder

    def forward(self,inp,encoder_input=None,encoder_output=None,encoder_attention_mask=None):
        """
        'inp': input to the decoder part
        'encoder_input': the input data point itself
        'encoder_output': the output of the encoder (if None, it will be calculated for you)
        'encoder_attention_mask': If None, it will be all-ones, ie whole encoded input can be seen
        """
        if encoder_output is None:
            if encoder_attention_mask is None:
                encoder_attention_mask=torch.ones_like(encoder_input)
            preds_t=self.encoder(encoder_input,attention_mask=encoder_attention_mask)
            encoder_output=preds_t[2][-1] #TODO! VERIFY! preds_t[2] is the hidden_state_output, and [-1] is the last encoder layer
        decoder_output=self.decoder(inp_classes,attention_mask=torch.ones_like(inp_classes),encoder_hidden_states=encoder_output,encoder_attention_mask=encoder_attention_mask)
        
    

    
