import os
import transformers
from torch import nn
import torch
import sys

transformers.BERT_PRETRAINED_MODEL_ARCHIVE_MAP["pbert-v1"]="proteiinipertti-v1/pytorch_model.bin"
transformers.BERT_PRETRAINED_CONFIG_ARCHIVE_MAP["pbert-v1"]="proteiinipertti-v1/config.json"
transformers.tokenization_bert.PRETRAINED_VOCAB_FILES_MAP["vocab_file"]["pbert-v1"]="proteiinipertti-v1/vocab.txt"
transformers.tokenization_bert.PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES["pbert-v1"]=512
transformers.tokenization_bert.PRETRAINED_INIT_CONFIGURATION["pbert-v1"]={'do_lower_case': False}


class MlabelSimple(nn.Module):

    @classmethod
    def from_cpoint(cls,f_name):
        d=torch.load(f_name)
        #bert=transformers.BertModel.from_pretrained("pbert-v1")
        bert=transformers.modeling_bert.BertModel.from_pretrained(pretrained_model_name_or_path=f_name.replace(".torch",".bert"))#,state_dict=d["classifier_state_dict"],config=d["bert_config"])
        bert.train()
        m=cls(bert,d["label_count"])
        m.classifier.load_state_dict(d["classifier_state_dict"])
        m.classifier=m.classifier.cuda()
        m.classifier.train()
        return m, d

    def cuda(self):
        self.encoder=self.encoder.cuda()
        self.classifier=self.classifier.cuda()
        return self
    
    def train(self):
        self.classifier.train()
        self.encoder.train()

    def eval(self):
        self.classifier.eval()
        self.encoder.eval()

    def __init__(self,bert,label_count):
        super().__init__()
        self.label_count=label_count
        self.encoder=bert
        self.classifier=nn.Linear(self.encoder.config.hidden_size,label_count)

    def forward(self,encoder_input,sigm=True):
        last_hidden,cls=self.encoder(encoder_input)
        if sigm:
            return torch.sigmoid(self.classifier(cls))
        else:
            return self.classifier(cls)
        #classification_output=self.classifier(bert_encoded

    def save(self,f_name,xtra_dict={}):
        os.makedirs(f_name.replace(".torch",".bert"),exist_ok=True)
        self.encoder.save_pretrained(f_name.replace(".torch",".bert"))
        d={"classifier_state_dict":self.classifier.state_dict(),
           "label_count":self.label_count}
        for k,v in xtra_dict.items():
            assert k not in d
            d[k]=v
        torch.save(d,f_name)
        
        
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
    

    
