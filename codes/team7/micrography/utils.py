from pathlib import Path
import h5py
import numpy as np
from tqdm import tqdm
import requests

def download(url, fname):
    r = requests.get(url, stream=True)
    with open(fname, 'wb') as f:
        total_length = int(r.headers.get('content-length'))
        for chunk in tqdm(r.iter_content(chunk_size=1024), total=total_length/1024, unit='KB'): 
            if chunk:
                f.write(chunk)
                f.flush()


def download_data():
    output_dir = Path('data')
    output_dir.mkdir(exist_ok=True)

    model_files = ['Sm_0_1_HAADF.h5',
                'Sm_0_1_UCParameterization.h5',
                'Sm_7_2_HAADF.h5',
                'Sm_7_2_UCParameterization.h5',
                'Sm_10_1_HAADF.h5',
                'Sm_10_1_UCParameterization.h5',
                'Sm_13_0_HAADF.h5',
                'Sm_13_0_UCParameterization.h5',
                'Sm_20_0_HAADF.h5',
                'Sm_20_0_UCParameterization.h5']

    for model_file in model_files:
        if not (output_dir / model_file).exists():
            print(f'Downloading: {output_dir / model_file}')
            download("https://zenodo.org/record/4555979/files/"+model_file+"?download=1", str(output_dir / model_file))


def map2grid(inab, inVal):
    """maps x,y grid positions into a matrix data format"""

    abrng = [
        int(np.min(inab[:,0])), int(np.max(inab[:,0])), 
        int(np.min(inab[:,1])), int(np.max(inab[:,1]))
    ]
    abind = inab.copy()
    abind[:,0] -= abrng[0]
    abind[:,1] -= abrng[2]
    grid = np.empty((abrng[1]-abrng[0]+1,abrng[3]-abrng[2]+1))
    grid[:] = np.nan
    grid[abind[:,0].astype(int),abind[:,1].astype(int)]=inVal[:]
    return grid, abrng


def parse_files():
    target = Path('data')
    img_files = sorted(list(target.glob("*HAADF.h5")))
    uc_param_files = sorted(list(target.glob("*UC*.h5")))

    compositions = [int(x.name.split("_")[1]) for x in img_files]
    uc_params = [h5py.File(x, 'r') for x in uc_param_files]
    imgdata = [h5py.File(x, 'r')['MainImage'] for x in img_files]
    return compositions, uc_params, imgdata


def process_data(compositions, uc_params, imgdata):
    SBFO_data = []     #this will be the output list of dictionaries for each dataset

    for i in np.arange(len(compositions)):
        temp_dict = {}
        temp_dict['composition'] = compositions[i]
        temp_dict['image'] = imgdata[i]

        for k in uc_params[i].keys():       #add labels for UC parameterization
            temp_dict[k] = uc_params[i][k][()]

        #select values mapped to ab grid
        temp_dict['ab_a'] = map2grid(uc_params[i]['ab'][()].T, uc_params[i]['ab'][()].T[:,0])[0]       #a array
        temp_dict['ab_b'] = map2grid(uc_params[i]['ab'][()].T, uc_params[i]['ab'][()].T[:,1])[0]       #b array
        temp_dict['ab_x'] = map2grid(uc_params[i]['ab'][()].T, uc_params[i]['xy_COM'][()].T[:,0])[0]   #x array
        temp_dict['ab_y'] = map2grid(uc_params[i]['ab'][()].T, uc_params[i]['xy_COM'][()].T[:,1])[0]   #y array
        temp_dict['ab_Px'] = map2grid(uc_params[i]['ab'][()].T, uc_params[i]['Pxy'][0])[0]             #Px array
        temp_dict['ab_Py'] = map2grid(uc_params[i]['ab'][()].T, uc_params[i]['Pxy'][1])[0]        #Py array
        temp_dict['vol'] = map2grid(uc_params[i]['ab'][()].T, uc_params[i]['Vol'])[0]     #Vol array

        SBFO_data.append(temp_dict)
    return SBFO_data