import numpy as np
import pandas as pd 
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import ShuffleSplit

# Imports Excel Data for MIC Vals
# Returns: (Antibiotic, Day, Strain) MIC values; (Treatment, Day, Strain) Days per treatment; (Days, Strain) Strain name
def import_data_MIC(file_path):
    antibiotics = ['PIP','TOB','CIP', 'LB']

    headers = ['Control_1', 'Control_2', 'Control_3', 'Control_4']
    for i in range(3):
        for j in range(4):
            for k in range(4):
                headers.append(antibiotics[i]+antibiotics[j]+'_'+str(k+1))

    df_PIPR = pd.read_excel(open(file_path[0],'rb'), sheet_name='DataForFig2andS4', skiprows=4, skipfooter=92, usecols='B:BA')
    

    df_TOBR = pd.read_excel(open(file_path[0],'rb'), sheet_name='DataForFig2andS4', skiprows=50, skipfooter=46, usecols='B:BA')
    

    df_CIPR = pd.read_excel(open(file_path[0],'rb'), sheet_name='DataForFig2andS4', skiprows=96, usecols='B:BA')
    
    MICs_unorderd = np.array([df_PIPR.to_numpy(), df_TOBR.to_numpy(), df_CIPR.to_numpy()])

    # re-order columns to make them easier to transverse through (give pattern: Control, PP, PT, PC, PLB, TP)
    idx_start = [0,1,4,6,10,5,2,7,11,9,8,3,12]
    idx = np.ndarray.flatten(np.array([np.arange(start*4, (start+1)*4) for start in idx_start]))
    MICs = MICs_unorderd[:, :, idx]

    # Fill in missing carry over values: e.g. first 20 days of PIPPIP = first 20 days of PIPTOB, PIPCIP, PIPLB
    MICs[:,0:20, 2*4:5*4] = np.tile(MICs[:,0:20, 1*4:2*4], 3)
    MICs[:,0:20, 7*4:9*4] = np.tile(MICs[:,0:20, 6*4:7*4],2)
    MICs[:,0:20, 5*4:6*4] = MICs[:,0:20, 6*4:7*4]
    MICs[:,0:20, 9*4:11*4] = np.tile(MICs[:,0:20, 11*4:12*4],2)
    MICs[:,0:20, 12*4:13*4] = MICs[:,0:20, 11*4:12*4]

    days = np.zeros(shape = (4, MICs.shape[1], MICs.shape[2]))
    recent = np.zeros(shape = (4, MICs.shape[1], MICs.shape[2]))

    strains = np.empty_like(MICs[0,:,:], dtype=object)
    for day in range(MICs.shape[1]):
        if day < 20:
            days[0, day, 4*1:4*5] = day+1
            recent[0, day, 4*1:4*5] = 1

            days[1, day, 4*5:4*9] = day+1
            recent[1, day, 4*5:4*9] = 1

            days[2, day, 4*9:4*13] = day+1
            recent[2, day, 4*9:4*13] = 1

            days[3, day, 4*0:4*1] = day+1
            recent[3, day, 4*0:4*1] = 1
        else:
            days[0, day, 4*1:4*5] = 20
            days[1, day, 4*5:4*9] = 20
            days[2, day, 4*9:4*13] = 20
            days[3, day, 4*0:4*1] = 20

            for i in [1,5,9]:

                days[0, day, 4*i:4*(i+1)] = (day+1)-20
                recent[0, day, 4*i:4*(i+1)] =  1

                days[1, day, 4*(i+1):4*(i+2)] = (day+1)-20
                recent[1, day, 4*(i+1):4*(i+2)] = 1

                days[2, day, 4*(i+2):4*(i+3)] = (day+1)-20
                recent[2, day, 4*(i+2):4*(i+3)] = 1
                
                days[3, day, 4*(i+3):4*(i+4)] = (day+1)-20
                recent[3, day, 4*(i+3):4*(i+4)] = 1
                

            days[0, day, 4*1:4*(1+1)] = (day+1)
            recent[0, day, 4*1:4*(1+1)] = 1

            days[1, day, 4*6:4*(6+1)] = (day+1)
            recent[1, day, 4*6:4*(6+1)] = 1

            days[2, day, 4*11:4*(11+1)] = (day+1)
            recent[2, day, 4*11:4*(11+1)] = 1

            days[3, day, 4*0:4*(0+1)] = (day+1)
            recent[3, day, 4*0:4*(0+1)] = 1

        for strain in range(MICs.shape[2]):
            strains[day, strain] = headers[strain] 

    # df_genes = pd.read_excel(open(file_path[1],'rb'), sheet_name='main_table', skiprows=3, usecols='M:BP')

    return MICs, days, strains, recent

def import_data_genes(file_path):
    return 0

# Reformats feature and data vectors
# Returns: (days*strain,antibiotic) MIC Vals, (days*strain,treatments) Treatment Days Feature, (days*strains, ) Strain Names, (days*strains, genes) Gene Mutation Feature
def preprocess(MICs,days,strains, recent):
    MICs_flat = MICs.reshape((MICs.shape[0], MICs.shape[1]*MICs.shape[2])).T
    days_flat = days.reshape((days.shape[0], days.shape[1]*days.shape[2])).T
    strains_flat = strains.flatten()
    recent_flat = recent.reshape((recent.shape[0], recent.shape[1]*recent.shape[2])).T

    return MICs_flat, days_flat, strains_flat, recent_flat
   
        
def main():
    file_path = ['../Paper_Data/s018.xlsx','../Paper_Data/mutations_table_outout_pretty.xlsx']

    MICs, days, strains, recent = import_data_MIC(file_path)



    np.save('../Processed_Data/MICs',MICs)
    np.save('../Processed_Data/Days',days)
    np.save('../Processed_Data/Strains',strains)
    np.save('../Processed_Data/Recent',recent)


    MICs_2, days_2, strains_2, recent_2 = preprocess(MICs,days, strains, recent)

    # Check to make sure indices are aligned
    # print(recent[:,19:25,36])
    # print(strains[19:25, 36])
    # print(days[:,19:25,36])
    
    np.save('../Processed_Data/MICs_flat',MICs_2)
    np.save('../Processed_Data/Days_flat',days_2)
    np.save('../Processed_Data/Strains_flat',strains_2)
    np.save('../Processed_Data/Recent_flat',recent_2)

  
if __name__== "__main__":
  main()