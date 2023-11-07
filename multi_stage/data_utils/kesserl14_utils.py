import os
import numpy as np
from data_utils.label_parse import LabelParser
from data_utils import shared_utils
from data_utils import current_program_code as cpc
from open_source_utils import stanford_utils
# from transformers import BertTokenizer
from transformers import AutoModel, AutoTokenizer


class DataGenerator(object):
    def __init__(self, config):
        """
        :param config: a program configure
        :return: input_ids, attn_mask, pos_ids, dep_matrix, dep_label_matrix, label_ids
        """
        self.config = config
        self.vocab, self.pos_dict = {"<pad>": 0, "<s>": 1, "</s>": 2}, {"<pad>": 0}
        self.vocab_index, self.pos_index = 5, 5
        self.token_max_len, self.char_max_len = -1, -1

        # store some data using in model
        self.train_data_dict, self.dev_data_dict, self.test_data_dict = {}, {}, {}
        self.bert_tokenizer = AutoTokenizer.from_pretrained(config.path.bert_model_path)

        self.elem_col = ["subject", "object", "aspect", "predicate"]

    def create_data_dict(self, data_path, data_type, label_path=None):
        """
        :param data_path: sentence file path
        :param data_type:
        :param label_path: label file path
        :return: a data dict with many parameters
        """
        data_dict = {}

        sent_col, sent_label_col, label_col = cpc.read_standard_file(data_path)
        if(data_type == 'test'): 
            data_dict['sentences'] = sent_col
        LP = LabelParser(label_col, ["subject", "object", "aspect", "predicate"])
        label_col, tuple_pair_col = LP.parse_sequence_label("&&", sent_col, file_type="en")

        # using stanford tool to get some feature data.
        if not os.path.exists(self.config.path.pre_process_data[data_type]):
            
            sf = stanford_utils.stanfordFeature(sent_col)
            
            data_dict['standard_token'] = sf.get_tokenizer()
            
            # shared_utils.write_pickle(data_dict, self.config.path.pre_process_data[data_type])
        else:
            data_dict = shared_utils.read_pickle(self.config.path.pre_process_data[data_type])

        self.token_max_len = max(self.token_max_len, shared_utils.get_max_token_length(data_dict['standard_token']))

        data_dict['label_col'] = label_col
        data_dict['comparative_label'] = sent_label_col

        if self.config.model_mode == "bert":
            data_dict['bert_token'] = shared_utils.get_token_col(sent_col, bert_tokenizer=self.bert_tokenizer, dim=1)

                # print('bert',data_dict['bert_token'][0])
                # print('stand',data_dict['standard_token'][0])
            mapping_col = shared_utils.token_mapping_bert(data_dict['bert_token'], data_dict['standard_token'])
            if(data_type == 'test'): 
                data_dict['mapping_col'] = mapping_col
            if(data_type == 'train'):
                
                label_col = cpc.convert_vi_label_dict_by_mapping(label_col, mapping_col)

                tuple_pair_col = cpc.convert_vi_tuple_pair_by_mapping(tuple_pair_col, mapping_col)

            data_dict['input_ids'] = shared_utils.bert_data_transfer(
                self.bert_tokenizer,
                data_dict['bert_token'],
                "tokens"
            )

            self.char_max_len = max(self.char_max_len, shared_utils.get_max_token_length(data_dict['input_ids'])) + 2

        else:

            self.vocab, self.vocab_index = shared_utils.update_vocab(
                data_dict['standard_token'],
                self.vocab,
                self.vocab_index,
                dim=2
            )

            data_dict['input_ids'] = shared_utils.transfer_data(data_dict['standard_token'], self.vocab, dim=1)

            self.char_max_len = max(self.char_max_len, shared_utils.get_max_token_length(data_dict['input_ids'])) + 2

        data_dict['tuple_pair_col'] = tuple_pair_col
        print("convert pair number: ", cpc.get_tuple_pair_num(data_dict['tuple_pair_col']))

        token_col = data_dict['standard_token'] if self.config.model_mode == "norm" else data_dict['bert_token']

        data_dict['attn_mask'] = shared_utils.get_mask(token_col, dim=1)

        special_symbol = False

        # multi-label: a sentence denote four sequence-label. [N, 3, sequence_length]
        # result_label: [N, sequence_length] polarity-col: [N, pair_num]
        data_dict['multi_label'], data_dict['result_label'], data_dict['polarity_label'] = \
            cpc.elem_dict_convert_to_multi_sequence_label(
                token_col, label_col, special_symbol=special_symbol
            )

        ################################################################################################################
        # tags to ids
        ################################################################################################################

        data_dict['multi_label'] = shared_utils.transfer_data(
            data_dict['multi_label'],
            self.config.val.norm_id_map,
            dim=2
        )

        data_dict['result_label'] = shared_utils.transfer_data(
            data_dict['result_label'],
            self.config.val.norm_id_map,
            dim=1
        )

        return data_dict

    def generate_data(self):
        self.train_data_dict = self.create_data_dict(
            self.config.path.standard_path['train'],
            "train"
        )

        self.dev_data_dict = self.create_data_dict(
            self.config.path.standard_path['dev'],
            "dev"
        )

        self.test_data_dict = self.create_data_dict(
            self.config.path.standard_path['test'],
            "test"
        )

        self.train_data_dict = self.padding_data_dict(self.train_data_dict)
        self.dev_data_dict = self.padding_data_dict(self.dev_data_dict)
        self.test_data_dict = self.padding_data_dict(self.test_data_dict)

        self.train_data_dict = self.data_dict_to_numpy(self.train_data_dict, 'train')
        self.dev_data_dict = self.data_dict_to_numpy(self.dev_data_dict, 'dev')
        self.test_data_dict = self.data_dict_to_numpy(self.test_data_dict, 'train')

    def padding_data_dict(self, data_dict):
        """
        :param data_dict:
        :return:
        """
        pad_key_ids = {0: ["input_ids", "attn_mask", "result_label"],
                       1: ["multi_label"]}

        cur_max_len = self.char_max_len

        param = [{"max_len": cur_max_len, "dim": 1, "pad_num": 0, "data_type": "norm"},
                 {"max_len": cur_max_len, "dim": 2, "pad_num": 0, "data_type": "norm"}]

        for index, key_col in pad_key_ids.items():
            for key in key_col:
                data_dict[key] = shared_utils.padding_data(
                    data_dict[key],
                    max_len=param[index]['max_len'],
                    dim=param[index]['dim'],
                    padding_num=param[index]['pad_num'],
                    data_type=param[index]['data_type']
                )

        return data_dict

    @staticmethod
    def data_dict_to_numpy(data_dict, _type):
        """
        :param data_dict:
        :return:
        """
        key_col = ["input_ids", "attn_mask", "tuple_pair_col", "result_label", "multi_label", "comparative_label"]
        if _type == 'test':
            key_col.insert(0, 'sentences')
            
        for key in key_col:
            data_dict[key] = np.array(data_dict[key])
            print(key, data_dict[key].shape)

        data_dict['comparative_label'] = np.array(data_dict['comparative_label']).reshape(-1, 1)

        return data_dict