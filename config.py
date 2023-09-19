
class transformer_config:
    def __init(self):
        self.transformer_config = {
            'prediction_length':7,
            'context_length':7, #same as prediction_length
            'distribution_output':'student_t',
            'utlize_default_lags_sequence':False,
            'lags_sequence': [1,2,3,4,5,6,7]
        }