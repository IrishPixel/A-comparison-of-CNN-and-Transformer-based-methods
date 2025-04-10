from multiprocessing.dummy import Pool as ThreadPool
from operator import itemgetter
import sys
import stainNorm_Macenko
import multiprocessing
import os
import cv2
import numpy  as np
from pathlib import Path
from itertools import repeat
import argparse
import glob
import pandas as pd
#TODO: create process bar that works in multiprocessing env
#from tqdm import tqdm

# global inputPath
# global outputPath
#global normalizer # remove normalizer as global variable

##############################################################################



def Normalize_Main(inputPath, outputPath, item, normalizer): 

    outputPathRoot = os.path.join(outputPath, item)
    inputPathRoot = os.path.join(inputPath, item)

    #check if path actually leads to a directory
    if(os.path.isdir(inputPathRoot)):
        inputPathRootContent = os.listdir(inputPathRoot)
        if not len(inputPathRootContent) == 0:
            if not os.path.exists(outputPathRoot):
                #creates the parent directory (patient) and child (file)
                if '/' in item: #parent/child structure from clini table input
                    os.makedirs(outputPathRoot, exist_ok=False)
                else:
                    os.mkdir(outputPathRoot)

                temp = os.path.join(inputPath, item)
                tempContent = os.listdir(temp)
                tempContent = [i for i in tempContent if i.endswith('.jpg') or ('.png')]
                for tempItem in tempContent:
                    img = cv2.imread(os.path.join(inputPathRoot, tempItem))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    edge  = cv2.Canny(img, 40, 100)
                    edge = edge / np.max(edge) if np.max(edge) != 0 else 0
                    edge = (np.sum(np.sum(edge)) / (img.shape[0] *img.shape[1])) * 100 if np.max(edge) != 0 else 0
                    #print(edge)
                    if edge > 2:
                        try:
                            nor_img = normalizer.transform(img)
                            cv2.imwrite(os.path.join(outputPathRoot, tempItem), cv2.cvtColor(nor_img, cv2.COLOR_RGB2BGR))
                        except:
                            print('Failed to normalize the tile {}.'.format(tempItem))
    else:
        print(f"Rejected {inputPathRoot}: not a directory")
                    
##############################################################################
                        
def poolcontext(*args, **kwargs):
    pool = multiprocessing.Pool(*args, **kwargs)
    yield pool
    pool.terminate()

###############################################################################
    
def Normalization(inputPath: Path, outputPath: Path, sampleImagePath: Path, num_threads: int, patient_list: Path) -> None:
    
    if patient_list == None:
        inputPathContent = os.listdir(inputPath)
    
    else:
        print("Using provided patient list to determine normalisation slides...")
        all_tiled_slides = glob.glob(f"{inputPath}/*/*") #this will contain all your subfolders of the patients

        #read the excel file which contains the names which we actually want
        df = pd.read_excel(patient_list, dtype=str)
        #extract the column we want from the table
        df=(df[['PATIENT','FILENAME']].dropna().values).tolist()
        patient_list=[f"{inputPath}/{patient}/{file}" for patient, file in df]

        #find intersection between the folder lists and our clini patient list
        #format into item object within normalisation function patient/filename
        inputPathContent = [f"{full_dir.split('/')[-2]}/{full_dir.split('/')[-1]}" \
            for full_dir in list(set(all_tiled_slides) & set(patient_list))]
        print(f"... Found {len(inputPathContent)} slides to normalise")
    
    normPathContent = os.listdir(outputPath)
    
    remainlList = []
    for i in inputPathContent:
        if not i in normPathContent:
            remainlList.append(i)
            
    inputPathContent = [i for i in remainlList if not i.endswith('.bat')]
    inputPathContent = [i for i in inputPathContent if not i.endswith('.txt')]
    
    target = cv2.imread(sampleImagePath)
    target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
        
    pool = ThreadPool(num_threads)
    normalizer = stainNorm_Macenko.Normalizer()
    normalizer.fit(target)
    # old function, only passed inputpath
    # pool.map(Normalize_Main, inputPathContent)
    
    #quick fix uses starmap, which passes the arguments as iterative objects.
    #repeat is used for constants
    pool.starmap(Normalize_Main, zip((repeat(inputPath)), repeat(outputPath), inputPathContent, repeat(normalizer)))
    pool.close()
    pool.join()

if __name__ == '__main__':
    #parsing all arguments from the command line
    parser = argparse.ArgumentParser()
    requiredNamed = parser.add_argument_group('required named arguments')
    requiredNamed.add_argument("-ip", "--inputPath", help="Input path of the to-be-normalised tiles", type=Path, required=True)
    requiredNamed.add_argument("-op", "--outputPath", help="Output path to store normalised tiles", type=Path, required=True)
    parser.add_argument("-pl", "--patientList", help="Clini table containing PATIENT and FILENAME to normalise", type=Path)
    parser.add_argument("-si", "--sampleImagePath", help="Image used to determine the colour distribution, uses GitHub one by default", type=Path)
    parser.add_argument("-nt", "--threads", help="Number of threads used for processing, 2 by default", type=int)
    args = parser.parse_args()

    #calling the Normalization function with defined parameters
    Normalization(  args.inputPath,
                    args.outputPath, 
                    args.sampleImagePath if args.sampleImagePath != None else 'normalization_template.jpg', 
                    args.threads if args.threads != None else 2,
                    args.patientList)
