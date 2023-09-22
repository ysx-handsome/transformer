
class transformer_config:
    def __init(self):
        self.transformer_config = {
            'input_size':1,
            'dec_seq_len':4,
            'dim_val':512,
            'n_encoder_layers':4,
            'n_decoder_layers':4,
            'n_heads':4,
            'dropout_encoder':0.2,
            'dropout_decoder':0.2,
            'dropout_pos_enc':0.1,
            'dim_feedforward_encoder':2048,
            'dim_feedforward_decoder':2048,
            'num_predicted_feature':1,
        }

        self.config = {
            'device':'cpu',
            'test_size':0.2,
            'batch_size':128,
            'target_col_name':'',
            'exogenous_vars':[]
        }
    