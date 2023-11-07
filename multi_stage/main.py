import torch
import json
import numpy as np
import re
import random
import os
import argparse
import Config

from data_utils import shared_utils, kesserl14_utils, data_loader_utils
from model_utils import train_test_utils
from eval_utils.base_eval import BaseEvaluation, ElementEvaluation, PairEvaluation
from eval_utils import create_eval
from data_utils import current_program_code as cpc


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.nn.Module.dump_patches = True


def TerminalParser():
    # define parse parameters
    parser = argparse.ArgumentParser()
    parser.description = 'choose train data and test data file path'

    parser.add_argument('--seed', help='random seed', type=int, default=2023)
    parser.add_argument('--batch', help='input data batch size', type=int, default=16)
    parser.add_argument('--epoch', help='the number of run times', type=int, default=10)
    parser.add_argument('--fold', help='the fold of data', type=int, default=5)

    # lstm parameters setting
    parser.add_argument('--input_size', help='the size of encoder embedding', type=int, default=300)
    parser.add_argument('--hidden_size', help='the size of hidden embedding', type=int, default=512)
    parser.add_argument('--num_layers', help='the number of layer', type=int, default=2)

    # program mode choose.
    parser.add_argument('--model_mode', help='bert or norm', default='bert')
    parser.add_argument('--server_type', help='1080ti or rtx', default='1080ti')
    parser.add_argument('--program_mode', help='debug or run or test', default='run')
    parser.add_argument('--stage_model', help='first or second', default='first')
    parser.add_argument('--model_type', help='bert_crf, bert_crf_mtl', default='crf')
    parser.add_argument('--position_sys', help='BIES or BI or SPAN', default='BMES')

    parser.add_argument('--device', help='run program in device type',
                        default='cuda' if torch.cuda.is_available() else 'cpu')

    parser.add_argument('--file_type', help='the type of data set', default='smartphone')
    parser.add_argument('--premodel_path', help='the type of data set', default=None)

    # model parameters.
    parser.add_argument('--embed_dropout', help='prob of embedding dropout', type=float, default=0.1)
    parser.add_argument('--factor', help='the type of data set', type=float, default=0.3)

    # optimizer parameters.
    parser.add_argument('--bert_lr', help='the type of data set', type=float, default=2e-5)
    parser.add_argument('--linear_lr', help='the type of data set', type=float, default=2e-5)
    parser.add_argument('--crf_lr', help='the type of data set', type=float, default=0.01)

    args = parser.parse_args()

    return args


def get_necessary_parameters(args):
    """
    :param args:
    :return:
    """
    param_dict = {"file_type": args.file_type,
                  "model_mode": args.model_mode,
                  "stage_model": args.stage_model,
                  "model_type": args.model_type,
                  "epoch": args.epoch,
                  "batch_size": args.batch,
                  "program_mode": args.program_mode}

    return param_dict

def convert_data(data_type,file_name = ''):
    folder_path = ''
    std_sents = []
    if data_type == 'train':
        folder_path = '../data/smartphone/VLSP2023_ComOM_training_v2'
        des_file = '../data/smartphone/train.txt'
    if data_type == 'test':
        folder_path = '../data/smartphone/VLSP2023_ComOM_testing_v3'
        des_file = '../data/smartphone/test.txt'
    if data_type == 'dev':
        folder_path = '../data/smartphone/VLSP2023_ComOM_dev_v2'
        des_file = '../data/smartphone/dev.txt'

    
    files = os.listdir(folder_path)
    if (data_type == 'test'):
        print(file_name)
        files.clear()
        files.append(file_name)
    sentences_and_content = []
    for file_name in files:
        # if file_name.startswith('train_') and file_name.endswith('.txt'):
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, 'r', encoding='utf-8') as file:
            sections = file.read().split('\n\n')
            for section in sections:
                tmp=""
                parts = section.split('\n')
                json_format = ""
                if len(parts) >= 2:
                    sentence = parts[0].strip()
                    tmp, sentence = sentence.split('\t')
                    sentence = " ".join(sentence.split())
                    sentence += '\t' + '1'

                    contents = parts[1].strip()
                    json_contents = contents.strip().split('\n')
                    combined_format = sentence + '\n'
                    for i in range(1, len(parts)):
                        for json_content in json_contents:
                            idx_s, idx_e = (-1, -1)
                            tuples = ""
                            while True:
                                idx_s = json_content.find('[', idx_s + 1)
                                idx_e = json_content.find(']', idx_e + 1)

                                if idx_s == -1:
                                    break

                                for i in json_content[idx_s:idx_e + 1]:
                                    if i != ',':
                                        tuples += i

                                tuples += ';'

                            labels = ["DIF", "EQL", "SUP+", "SUP-", "SUP", "COM+", "COM-", "COM"]

                            for label in labels:
                                if json_content.find(label) != -1:
                                    if label == "DIF":
                                        tuples += '[' + str(-1) + ']'
                                    elif label == "EQL":
                                        tuples += '[' + str(0) + ']'
                                    elif label == "SUP+":
                                        tuples += '[' + str(1) + ']'
                                    elif label == "SUP-":
                                        tuples += '[' + str(2) + ']'
                                    elif label == "SUP":
                                        tuples += '[' + str(3) + ']'
                                    elif label == "COM+":
                                        tuples += '[' + str(4) + ']'
                                    elif label == "COM-":
                                        tuples += '[' + str(5) + ']'
                                    elif label == "COM":
                                        tuples += '[' + str(6) + ']'
                                    break

                            json_format += '[' + tuples + ']' + '\n'
                            json_format = json_format.replace('"', '')
                    combined_format += json_format[:-1]
                    sentences_and_content.append(combined_format)

                else:
                    sentence = parts[0].split('\t')
                    if len(sentence) == 1:
                        continue
                    else:
                        tmp = sentence[0]
                        sentence = sentence[-1]
                    sentence = " ".join(sentence.split())
                    sentence += '\t' + '0'
                    json_format = "[[];[];[];[];[]]"
                    combined_format = f"{sentence}\n{json_format}"
                    sentences_and_content.append(combined_format)
                if(data_type == 'test'):
                    
                    std_sents.append(tmp)
    
    with open(des_file, 'w', encoding='utf-8') as output_file:
        for item in sentences_and_content:
            output_file.write(str(item) + '\n')
    return std_sents
            
def main():    
    # get program configure
    args = TerminalParser()
    # set random seed
    set_seed(args.seed)

    config = Config.BaseConfig(args)
    config_parameters = get_necessary_parameters(args)

    if args.stage_model == "first":
        model_parameters = {"embed_dropout": args.embed_dropout}
    else:
        model_parameters = {"embed_dropout": args.embed_dropout, "factor": args.factor}

    optimizer_parameters = None

    model_name = shared_utils.parameters_to_model_name(
        {"config": config_parameters, "model": model_parameters}
    )

    print('model_name',model_name)

    # if config.data_type == "eng":
    #     data_gene = kesserl14_utils.DataGenerator(config)
    # else:
    #     data_gene = coae13_utils.DataGenerator(config)
    
    s = convert_data('train')
    
    a = convert_data('dev')
    for i in range(1,37):
      if(i>=10):
        filename = f"test_00{i}.txt"
      else:
        filename = f"test_000{i}.txt"

      std_sents = convert_data('test',file_name = filename)
      data_gene = kesserl14_utils.DataGenerator(config)
      data_gene.generate_data()

      data_gene.test_data_dict['std_sents'] = std_sents
      data_gene.test_data_dict['file_name'] = filename
      print('len', len(data_gene.test_data_dict['sentences']))
      global_eval = BaseEvaluation(config)
      global_pair_eval = BaseEvaluation(config)

      print("create data loader")
      train_loader = data_loader_utils.create_first_data_loader(
          data_gene.train_data_dict, config.batch_size
      )

      dev_loader = data_loader_utils.create_first_data_loader(
          data_gene.dev_data_dict, config.batch_size
      )

      test_loader = data_loader_utils.create_first_data_loader(
          data_gene.test_data_dict, config.batch_size
      )

      # run first-stage model.(extract four type elements)
      if config.stage_model == "first" and config.program_mode != "test":
          first_data_loader = [train_loader, dev_loader, test_loader]

          dev_comp_eval = create_eval.create_first_stage_eval(
              config,
              (data_gene.dev_data_dict['multi_label'], data_gene.dev_data_dict['result_label']),
              data_gene.dev_data_dict['comparative_label'],
              data_gene.dev_data_dict['attn_mask'],
              save_model=True
          )

          test_comp_eval = create_eval.create_first_stage_eval(
              config,
              (data_gene.test_data_dict['multi_label'], data_gene.test_data_dict['result_label']),
              data_gene.test_data_dict['comparative_label'],
              data_gene.test_data_dict['attn_mask'],
              save_model=False
          )

          comp_eval = [dev_comp_eval, test_comp_eval, global_eval]

          train_test_utils.first_stage_model_main(
              config, data_gene, first_data_loader, comp_eval,
              model_parameters, optimizer_parameters,
              model_name
          )

      elif config.program_mode == "test" and config.stage_model == "first":
          dev_parameters = ["./ModelResult//" + model_name + "/dev_elem_result.txt",
                            "./PreTrainModel/" + model_name + "/dev_model"]

          print("========================test==========================")
          predicate_model = torch.load(dev_parameters[1])

          test_parameters = ["./ModelResult/" + model_name + "/test_elem_result.txt", None]

          test_comp_eval = create_eval.create_first_stage_eval(
              config,
              (data_gene.test_data_dict['multi_label'], data_gene.test_data_dict['result_label']),
              data_gene.test_data_dict['comparative_label'],
              data_gene.test_data_dict['attn_mask'],
              save_model=False
          )

          train_test_utils.first_stage_model_test(
              predicate_model, config, test_loader, test_comp_eval, test_parameters
          )

          test_comp_eval.print_elem_result(
              data_gene.test_data_dict['sentences'], data_gene.test_data_dict['input_ids'], data_gene.test_data_dict['attn_mask'],
              "./ModelResult/" + model_name + "/test_result_file" + ".txt", drop_span=False
          )

          # add average measure.
          shared_utils.calculate_average_measure(test_comp_eval, global_eval)

          #Stage 2
          feature_type = 0

          # using evaluation to generate index col and pair label.
          generate_second_res_eval = ElementEvaluation(
              config, elem_col=config.val.elem_col,
              ids_to_tags=config.val.invert_norm_id_map
          )
          

          pre_train_model_path = "./PreTrainModel/" + model_name + "/dev_model"

          if not os.path.exists(pre_train_model_path):
              print("[ERROR] pre-train model isn't exist")
              return

          elem_model = torch.load(pre_train_model_path)

          test_first_process_data_path = "./ModelResult/" + model_name + "/test_first_data_" + str(feature_type) + ".txt"

          # if os.path.exists(test_first_process_data_path):
          #     test_candidate_pair_col, test_pair_representation, test_make_pair_label = \
          #         shared_utils.read_pickle(test_first_process_data_path)

          # else:
          test_candidate_pair_col, test_pair_representation, test_make_pair_label, _, _ = \
              train_test_utils.first_stage_model_test(
                  elem_model, config, test_loader, generate_second_res_eval,
                  eval_parameters=[data_gene.test_data_dict['tuple_pair_col']],
                  test_type="gene", feature_type=feature_type
              )

          # shared_utils.write_pickle(
          #     [test_candidate_pair_col, test_pair_representation, test_make_pair_label],
          #     test_first_process_data_path
          # )

          dev_pair_parameters = ["./ModelResult/" + model_name + "/dev_pair_result.txt",
                                "./PreTrainModel/" + model_name + "/dev_pair_model"]

          dev_polarity_parameters = ["./ModelResult/" + model_name + "/dev_polarity_result.txt",
                                    "./PreTrainModel/" + model_name + "/dev_polarity_model"]

          test_pair_parameters = ["./ModelResult/" + model_name + "/test_pair_result.txt", None]
          test_polarity_parameters = ["./ModelResult/" + model_name + "/test_pair_result.txt", None]

          predict_pair_model = torch.load(dev_pair_parameters[1])
          predict_polarity_model = torch.load(dev_polarity_parameters[1])

          test_pair_eval = PairEvaluation(
              config,
              gold_pair_col=data_gene.test_data_dict['tuple_pair_col'],
              candidate_pair_col=test_candidate_pair_col,
              elem_col=config.val.elem_col,
              ids_to_tags=config.val.norm_id_map,
              save_model=False
          )

          test_pair_loader = data_loader_utils.get_loader([test_pair_representation], 1)

          # train_test_utils.pair_stage_model_test(
          #     predict_pair_model, config, test_pair_loader, test_pair_eval,
          #     test_pair_parameters, mode="pair", polarity=False, initialize=(False, False)
          # )
          shared_utils.calculate_average_measure(test_pair_eval, global_pair_eval)
          global_pair_eval.avg_model("./ModelResult/" + model_name + "/test_pair_result.txt")
          global_pair_eval.store_result_to_csv([model_name], "result.csv")

          shared_utils.clear_global_measure(global_pair_eval)
          shared_utils.clear_optimize_measure(test_pair_eval)

          # create polarity representation and data loader.
          test_polarity_representation = cpc.get_after_pair_representation(test_pair_eval.y_hat, test_pair_representation)
          test_polarity_loader = data_loader_utils.get_loader([test_polarity_representation], 1)

          train_test_utils.pair_stage_model_test(
              predict_polarity_model, config, test_polarity_loader, test_pair_eval,
              test_polarity_parameters, mode="polarity", polarity=True, initialize=(True, True),
              data_dict=data_gene.test_data_dict
          )

          # add average measure.
          shared_utils.calculate_average_measure(test_pair_eval, global_pair_eval)

      elif config.stage_model == "second":
          # 0: 768 + 5, 1: 5, 2: 768
          feature_type = 0

          # using evaluation to generate index col and pair label.
          generate_second_res_eval = ElementEvaluation(
              config, elem_col=config.val.elem_col,
              ids_to_tags=config.val.invert_norm_id_map
          )

          pre_train_model_path = "./PreTrainModel/" + model_name + "/dev_model"

          if not os.path.exists(pre_train_model_path):
              print("[ERROR] pre-train model isn't exist")
              return

          elem_model = torch.load(pre_train_model_path)

          print(model_name)
          print('before first process data')
          train_first_process_data_path = "./ModelResult/" + model_name + "/train_first_data_" + str(feature_type) + ".txt"
          dev_first_process_data_path = "./ModelResult/" + model_name + "/dev_first_data_" + str(feature_type) + ".txt"
          test_first_process_data_path = "./ModelResult/" + model_name + "/test_first_data_" + str(feature_type) + ".txt"

          # print(data_gene.train_data_dict['tuple_pair_col'])
          
          if os.path.exists(train_first_process_data_path):
              train_pair_representation, train_make_pair_label, train_polarity_representation, train_polarity_label = \
                  shared_utils.read_pickle(train_first_process_data_path)
          else:
              _, train_pair_representation, train_make_pair_label, train_feature_out, train_bert_feature_out = \
                  train_test_utils.first_stage_model_test(
                      elem_model, config, train_loader, generate_second_res_eval,
                      eval_parameters=[data_gene.train_data_dict['tuple_pair_col']],
                      test_type="gene", feature_type=feature_type
                  )
          
          # print(type(train_pair_representation), type(train_make_pair_label))
          train_pair_representation, train_make_pair_label = cpc.generate_train_pair_data(
              train_pair_representation, train_make_pair_label
          )

          train_polarity_representation, train_polarity_label = cpc.create_polarity_train_data(
              config, data_gene.train_data_dict['tuple_pair_col'], train_feature_out,
              train_bert_feature_out, feature_type=feature_type
          )

          shared_utils.write_pickle(
              [train_pair_representation, train_make_pair_label,
                  train_polarity_representation, train_polarity_label],
              train_first_process_data_path
          )

          if os.path.exists(dev_first_process_data_path):
              dev_candidate_pair_col, dev_pair_representation, dev_make_pair_label = \
                  shared_utils.read_pickle(dev_first_process_data_path)
              print('dev_candidate_pair_col', dev_candidate_pair_col)
              print('dev_make_pair_label', dev_make_pair_label)

          else:
              dev_candidate_pair_col, dev_pair_representation, dev_make_pair_label, _, _ = \
                  train_test_utils.first_stage_model_test(
                      elem_model, config, dev_loader, generate_second_res_eval,
                      eval_parameters=[data_gene.dev_data_dict['tuple_pair_col']],
                      test_type="gene", feature_type=feature_type
                  )

              shared_utils.write_pickle(
                  [dev_candidate_pair_col, dev_pair_representation, dev_make_pair_label],
                  dev_first_process_data_path
              )

          # print(dev_candidate_pair_col, '\n', dev_pair_representation, '\n', dev_make_pair_label)
          if os.path.exists(test_first_process_data_path):
              test_candidate_pair_col, test_pair_representation, test_make_pair_label = \
                  shared_utils.read_pickle(test_first_process_data_path)

          else:
              test_candidate_pair_col, test_pair_representation, test_make_pair_label, _, _ = \
                  train_test_utils.first_stage_model_test(
                      elem_model, config, test_loader, generate_second_res_eval,
                      eval_parameters=[data_gene.test_data_dict['tuple_pair_col']],
                      test_type="gene", feature_type=feature_type
                  )

              shared_utils.write_pickle(
                  [test_candidate_pair_col, test_pair_representation, test_make_pair_label],
                  test_first_process_data_path
              )

          pair_representation = [train_pair_representation, dev_pair_representation, test_pair_representation]
          make_pair_label = [train_make_pair_label, dev_make_pair_label, test_make_pair_label]

          dev_pair_eval = PairEvaluation(
              config,
              gold_pair_col=data_gene.dev_data_dict['tuple_pair_col'],
              candidate_pair_col=dev_candidate_pair_col,
              elem_col=config.val.elem_col,
              ids_to_tags=config.val.norm_id_map,
              save_model=True
          )

          test_pair_eval = PairEvaluation(
              config,
              gold_pair_col=data_gene.test_data_dict['tuple_pair_col'],
              candidate_pair_col=test_candidate_pair_col,
              elem_col=config.val.elem_col,
              ids_to_tags=config.val.norm_id_map,
              save_model=False
          )

          train_test_utils.pair_stage_model_main(
              config, pair_representation, make_pair_label,
              [dev_pair_eval, test_pair_eval, global_pair_eval],
              [train_polarity_representation, train_polarity_label],
              model_parameters, optimizer_parameters, model_name, feature_type
          )
          print(test_pair_eval.polarity_hat)
      
      if config.stage_model == "first":
          global_eval.avg_model("./ModelResult/" + model_name + "/test_extraction_result.txt")
          global_eval.store_result_to_csv([model_name], "result.csv")
      else:
          global_pair_eval.avg_model("./ModelResult/" + model_name + "/test_pair_result.txt")
          global_pair_eval.store_result_to_csv([model_name], "result.csv")


if __name__ == "__main__":
    main()