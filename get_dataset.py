import kaggle.api
import os
import argparse

argParser = argparse.ArgumentParser()
argParser.add_argument('-p', help = 'Caminho do dataset no Kaggle')
args = argParser.parse_args()

def get_dataset():
    
    """
    Faz o download de um dataset do Kaggle e extrai os arquivos para um diretório local.

    A função cria um diretório chamado 'data' (caso não exista) para armazenar os arquivos do dataset.
    Em seguida, autentica na API do Kaggle, faz o download e extrai os arquivos do dataset especificado
    no argumento '-p' da linha de comando para o diretório 'data'.

    Argumentos da linha de comando:
    -p: String, o caminho do dataset no Kaggle. Deve ser fornecido ao chamar o script.

    Exceções:
    A função captura exceções gerais e imprime uma mensagem de erro caso ocorra algum problema durante
    o processo de download ou extração dos arquivos.
    """
    
    dataset_path = 'data/'
    os.makedirs(dataset_path, exist_ok = True)        
    kaggle_path = str(args.p)
        
    try:
        
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(dataset = kaggle_path, 
                                          path = dataset_path, 
                                          unzip = True)
        
    except:
        
        print('Não possível extrair os dados')
    
if __name__ == '__main__':
    
    get_dataset()