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




def laminin_channel_gdrive_():

    laminin_IDs = ["1EsSFqzaph5d_WKHKwMpzkGXW3jpeAs8B","1jx1422AMlUdPvIx_o_l56Xr7JRWQUTD-"]
    laminin_output_names = ["10 Count 384.jpg", "131 count 351.jpg"]

    for ID, output_name in zip(laminin_IDs, laminin_output_names):
        download_file_from_google_drive(ID, output_name)
    return










"""
Loading the heavy model files ...
"""

def model_output_names(Phase_num):
    output_names = [f'Phase{Phase_num}_Kfold-model_0.h5', f'Phase{Phase_num}_Kfold-model_0.json',
                            f'Phase{Phase_num}_Kfold-model_1.h5', f'Phase{Phase_num}_Kfold-model_1.json',
                            f'Phase{Phase_num}_Kfold-model_2.h5', f'Phase{Phase_num}_Kfold-model_2.json',
                            f'Phase{Phase_num}_Kfold-model_3.h5', f'Phase{Phase_num}_Kfold-model_3.json',
                            f'Phase{Phase_num}_Kfold-model_4.h5', f'Phase{Phase_num}_Kfold-model_4.json']
    return output_names





def resnet50_gdrive_(Phase_num = '2'):

   
    resnet50_IDs = ["1G4LfnQQzn0ufnJkWvpEGmbdqXCZ02YKn", "1P16p6KZt04chEt8griUQc-A6n7f4H6O1",
    "1znKMg6KqkNGniIRtL6d3qEV41ZAJWrG4", "15qEOuYxGMPi4chUa1I9AMC7wM5slHUhm",
    "12JCTv0B_YLcU_fnAjScakeKQLaG28YPJ", "167KwcwxIejRdpwVL43652SR6B7M49jb3", 
    "1VRbQbMv4Jpa5eJ4H2rB_RVLMwqzmbKv2", "1gK3E5s3pSBlcjZ_ZKXk0c0f1C5rhZbaA",
    "1S9XwLXdI5n1Qq0EN5Kji8UOSruFIzsy1", "1TsJLWgXukfqEz5IMJZ0baNU-clKeqV8g"]
    resnet50_output_names = model_output_names(Phase_num = '2')
    
    print("Start downloading resnet50 net models...")
    for ID, output_name in zip(resnet50_IDs, resnet50_output_names):
        download_file_from_google_drive(ID, output_name)
    
    return 

def Xception_gdrive_(Phase_num = '2'):
    
    Xception_IDs = ["1BdZR-Y5zIKYMhDN0m9AadcO1GHnXvNh_","14jIJKcv8651i-SEHQWX9HhZFXwSF93Lu",
    "1E52yKTiVq4wSrY3W4K11COFW0kQPvXjN","1jc4019dxK6vY4OU1BtmHgqb79qtWoMmG",
    "1OrOQFKEafBfJpouzWDBNOK5ka7rvGxeR","1-upMajCfF7iGVGeMZliYnisPtpA845CZ",
    "1KcpaMzG0VNOLC5Vdy6YQVzkCUFWjbXdZv","1G6TdpoyhIEq0FgdEWXqVxds2jbhY6soF",
    "143OgDHMv2X3A0cfGteavpdeuGfmhtAFX","1VNkCObXgeU_Z7Z2wJNjgNtKSCnpPM-qx"]
    
    Xception_output_names = model_output_names(Phase_num = '2')
    
    print("Start downloading Xception net models...")
    for ID, output_name in zip(Xception_IDs, Xception_output_names):
        download_file_from_google_drive(ID, output_name)
    return 




"""
gdown is an easily implementation to download relative small files that are below 40E3 kb

"""


def Vanilla_gdrive_(Phase_num = 2):

    url='https://drive.google.com/uc?id='

    Vanilla_Ids = ["1eGgWi0x8Z9bMO-gTngDOGH13kdV-z0pO","1UJbnHzsUUX5kdzplIhXSKyBRejtMnCro",
	"1GlLb30Stojjv3RH04vJuCFZv_qGQitHj","1fetL0S2ruCt7Aac6aNGCbth6R_VALCjX",
	"1UR6ujxdz5bG9buu6tdnV0IQYPXbeIigj", "1W7T1krKzzHMXIiivOCQr_2ObPcjuUtpX",
	"1UTuquYQF-c-9EzyXIqiKOZ678JVC1taZ", "1U2_AzByN7gmwctI9NM_1oNc-Ci_CJR7M",
	"10udGB144HwYkiXdzi0L91TfmpqeggToI", "16xmsxPKV22mnXCaFN6nllevY8_0_5qgU"]

    Vanilla_output_names = model_output_names(Phase_num = '2')
	
    print("Start downloading Vanilla net models...")
    for ID, output_name in zip(Vanilla_Ids, Vanilla_output_names):
        gdown.download(url+ID, f"./{output_name}")
    return 



def ce_jaccard_gdrive_(Phase_num = '1'):
    
    url = 'https://drive.google.com/uc?id='

    ce_jaccard_Ids = ["1mtewaKMXqdIVmWj1mXKQfYK9Tr5OCD7J", "1lQHqgcRCDhmAL5BrxSyfYwtejmpt_rXP",
    "1K1cduESQHDozHZYInEE0PmREdvvahiRs", "1rwJ9PKUuZAw9ZON9u6P7KC9H05jIT1nZ",
    "1Tn0e3OPhmFjZXLksP2reSTls3CFwoXSv", "1RrhIiRtWYFGJTpq-IRTXYyldaO0fKXFN",
    "1-R1kK7Z89XVSSpraTS2DfT8nFXqek8Ar", "127f8DRvLuJ33V5_i-TtOZ4-SKrPzsCsT",
    "1EWPauaixDCOHcDdN8LyBYj_CrtgCn6MX", "1N2e2sNJ8agWAiEX9Fw4gjEwk5OcFo_RY"]
    
    ce_jaccard_output_names =  model_output_names(Phase_num = '1') 
    print("Start downloading encoder-decoder Ce-Jaccard net models...")
    for ID, output_name in zip(ce_jaccard_Ids, ce_jaccard_output_names):
        gdown.download(url+ID, f"./{output_name}")
    
    return

def hrnet_CeJaccard_gdrive_(Phase_num = '1'):
    
    url = 'https://drive.google.com/uc?id='

    hrnet_ce_jaccard_Ids = ["1F1znTDI0MJacZkJmmmteiRmxh2o7yDho", "1VK6PQaxLXvKasoKTmfJxFHFlgMHaiIJz",
    "1TJ2or6b_3E7wsLjAjx5fNDDzXR3yLNOQ", "1XHpbpF2RQ_kAyaO1ngA7mjCyk4Hy4ceX",
    "17Z4PUVIVxaiC_SkwthJq9BTWPOEVXF4w", "1L537FbWX5jrHlHd0K8OYDZt0r9q6h6Vj",
    "1qTxpeP8xYFKPYHMDaM6fvkxVIFqY7bPe", "1OXfE6Z0WbEe8qN1m5Yf1AyzQVCzRv4uH",
    "1XDQtbv9n5a0m8x3SQbyK0ofpjSdnSWVO", "1l5PoudTUeJpRnPb1PKWPwgSGONbzSme3"]
 
    hrnet_ce_jaccard_output_names =  model_output_names(Phase_num = '1') 
    print("Start downloading hr'net Ce-Jaccard net models...")
    for ID, output_name in zip(hrnet_ce_jaccard_Ids, hrnet_ce_jaccard_output_names):
        gdown.download(url+ID, f"./{output_name}")
    
    return
