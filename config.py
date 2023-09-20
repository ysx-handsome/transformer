
class transformer_config:
    def __init(self):
        self.transformer_config = {
            'prediction_length':24,
            'context_length':24*2, #same as prediction_length
            'distribution_output':'student_t',
            'utlize_default_lags_sequence':False,
            'lags_sequence': [1,2,3,4,5,6,7],
            'num_time_features':2, #year and month
            'num_static_categorical_features':2, # 油品编码，油站编码，
            'cardinality': -1,#待确认
            'embedding_dimension':32,
            'd_model':64,
            'encoder_layers':2,
            'decoder_layers':2,
            'encoder_attention_heads:':2,
            'decoder_attention_heads' :2
        }