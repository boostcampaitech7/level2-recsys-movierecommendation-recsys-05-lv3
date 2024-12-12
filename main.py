import argparse
import ast
from omegaconf import OmegaConf

import importlib


def main(args):

    print(f'--------------- INIT {args.model} ---------------')
    model_path = f'src.{args.model}.run'
    
    print(args.model,"모델 하시는 거 맞죠?")
    module = importlib.import_module(model_path)
    model = getattr(module, 'main')
    model(args)
    print('test')


    # subprocess.run([
    #     'python', 'src/{args.model_args}/run.py', args
    # ])

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
        choices=['ELECTRA','VGGNet','RoBERTa','ResNet','CLIP'],
        help='학습 및 예측할 모델을 선택할 수 있습니다.')
    arg('--device', '-d', '--d', type=str, 
        choices=['cuda', 'cpu', 'mps'], help='사용할 디바이스를 선택할 수 있습니다.')


    args = parser.parse_args()

    ######################## Config with yaml
    config_args = OmegaConf.create(vars(args))
    config_yaml = OmegaConf.load(args.config) if args.config else OmegaConf.create()

    # args에 있는 값이 config_yaml에 있는 값보다 우선함. (단, None이 아닌 값일 경우)
    for key in config_args.keys():
        if config_args[key] is not None:
            config_yaml[key] = config_args[key]
    
    # Configuration 콘솔에 출력
    print(OmegaConf.to_yaml(config_yaml))

    ######################## MAIN
    main(config_yaml)