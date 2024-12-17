import argparse
import importlib
import sys

from omegaconf import OmegaConf

def main(args):
    print(f'--------------- INIT {args.model} ---------------')
    model_path = f'src.{args.model}.run'
    
    print(args.model,"모델 하시는 거 맞죠?")
    module = importlib.import_module(model_path)
    model = getattr(module, 'main')
    model(args)

    ######################## SAVE PREDICT
    print(f'--------------- SAVE {args.model} PREDICT ---------------')
    



if __name__ == "__main__":


    ######################## BASIC ENVIRONMENT SETUP
    parser = argparse.ArgumentParser(description='parser')

    arg = parser.add_argument
    str2dict = lambda x: {k:int(v) for k,v in (i.split(':') for i in x.split(','))}

    # add basic arguments (no default value)
    arg('--config', '-c', '--c', type=str, 
        help='Configuration 파일을 설정합니다.', default="./config/model_config.yaml")
    arg('--model', '-m', '--m', type=str, 
        choices=['ADMMSLIM', 'BERT4Rec', 'CDAE', 'DeepFM', 'EASE',
                  'EASER', 'FM', 'LightGCN', 'MultiVAE', 'NCF', 'SASRec'],
        help='학습 및 예측할 모델을 선택할 수 있습니다.')
    arg('--device', '-d', '--d', type=str, 
        choices=['cuda', 'cpu', 'mps'], help='사용할 디바이스를 선택할 수 있습니다.')
    arg('--params', '-p', type=str, nargs='*',
        help="Override할 파라미터를 문자열로 입력해주세요.")

    args = parser.parse_args()
    
    ######################## Config with yaml
    config_args = OmegaConf.create(vars(args))
    config_yaml = OmegaConf.load(args.config) if args.config else OmegaConf.create()


    if config_args.model :
        config_yaml.model = config_args.model

    if hasattr(config_yaml, 'model_args') and hasattr(config_yaml.model_args, config_yaml.model):
        config_yaml.model_args = config_yaml.model_args[config_yaml.model]
    else:
        raise ValueError(f"Invalid model specified or missing model_args for '{config_yaml.model}'.")

    # args에 있는 값이 config_yaml에 있는 값보다 우선함. (단, None이 아닌 값일 경우)
    for key in config_args.keys():
        if config_args[key] is not None:
            config_yaml[key] = config_args[key]    
 
    params = config_yaml.pop('params','')
    
    print(OmegaConf.to_yaml(config_yaml))

    try:
        if len(params) % 2 != 0 :
            raise IndexError('파라미터 수가 맞지 않습니다.')
            sys.exit(1)

        result_dict = {}
        for i in range(0, len(params), 2):
            key = params[i]
            value = params[i+1]
            
            # value가 숫자인지 체크 후 변환
            if value.isdigit():
                value = int(value)  # 숫자라면 int로 변환
            # 아니면 그대로 문자열로 처리
            result_dict[key] = value

    except ValueError as e:
        print(e)
        sys.exit(1)

    try:
        if not set(result_dict.keys()).issubset(config_yaml.model_args.keys()) :
            raise ValueError('없는 파라미터가 입력되었습니다.')
            sys.exit(1)
    except ValueError as e:
        print(e)
        sys.exit(1)
        
    for x in result_dict.keys() :
        config_yaml.model_args[x] = result_dict[x]

    # Configuration 콘솔에 출력
    print(OmegaConf.to_yaml(config_yaml))

    ######################## MAIN
    main(config_yaml)