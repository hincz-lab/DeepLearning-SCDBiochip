# -*- coding: utf-8 -*-
import requests
import gdown


"""
Following functions 'download_file_from_google_drive(), get_confirm_token(), and save_response_content()'
was adapted from https://stackoverflow.com/a/39225039. Hat tip to both turdus-merula and tttthomasssss.
These functions are also used only for larger files > 40E3 kb 
"""

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

"""
Loading the heavy model files ...
"""

def model_output_names(Phase_num):
    if Phase_num == 2:
        output_names = [f'Phase{Phase_num}_Kfold-model_0.h5',
                                f'Phase{Phase_num}_Kfold-model_1.h5',
                                f'Phase{Phase_num}_Kfold-model_2.h5',
                                f'Phase{Phase_num}_Kfold-model_3.h5',
                                f'Phase{Phase_num}_Kfold-model_4.h5',]
    elif Phase_num == 1:
            output_names = [f'Phase{Phase_num}_Kfold-model_0.h5',
                                f'Phase{Phase_num}_Kfold-model_1.h5',
                                f'Phase{Phase_num}_Kfold-model_2.h5',
                                f'Phase{Phase_num}_Kfold-model_3.h5',
                                f'Phase{Phase_num}_Kfold-model_4.h5',
                                f'Phase{Phase_num}_Kfold-model_5.h5',
                                f'Phase{Phase_num}_Kfold-model_6.h5',]
    return output_names


def resnet50_gdrive_(Phase_num = '2'):

   
    resnet50_IDs = ["1TKXziY3GAxGzR9WFiNPN1sUKh8GJ2sFr","1b0U-3Pn89v3iTFsR5yujv5ujPC1BNGsN",
                    "1_W88ALOLfOFJRksqvFJ2Woy0UunUxuAJ","1Qr6PHGhc1QX1FT1F1386GGAeGjyeUmDG",
                    "1x2Mg_54XcRfFOsCHL3-B7mOCOvoYzuYA"]
    resnet50_output_names = model_output_names(Phase_num = '2')
    
    print("Start downloading resnet50 net models...")
    for ID, output_name in zip(resnet50_IDs, resnet50_output_names):
        download_file_from_google_drive(ID, output_name)
    
    return 

"""
gdown is an easily implementation to download relative small files that are below 40E3 kb
"""


def ce_jaccard_gdrive_(Phase_num = '1'):
    
    url = 'https://drive.google.com/uc?id='

    ce_jaccard_Ids = ["1-nviMbneGU7SjySL5aGUBKzdagwX8rfY","1FsEnOen0AsrL5au6Bfnzd94opn7y3YaH",
                      "1xAi0fdOfw3uluPX6OrdGc4w9bVR7cD9_","1Frt4S2p0KdmOYQ4UtKJIWfyx7_eLQAyX",
                      "1fxVfqUSipFn01hM4qDtwjXVBwU6-dVgh","1TP9lxs2ldTu1_y0g4AmFKcOwkw8-Btre",
                      "1RuJtOQasUPzvM_8wytNebbWb6XeqHv6Z"]
    
    ce_jaccard_output_names =  model_output_names(Phase_num = '1') 
    print("Start downloading encoder-decoder Ce-Jaccard net models...")
    for ID, output_name in zip(ce_jaccard_Ids, ce_jaccard_output_names):
        gdown.download(url+ID, f"./{output_name}", quiet = False)
    
    return
