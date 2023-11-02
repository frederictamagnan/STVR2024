import json
from pprint import pprint
from math import inf
from collections import defaultdict

from pprint import pprint

import json
import numpy as np
from math import sqrt



def normalize_one(x,in_min,in_max,out_min,out_max):
    print(x)
    return (x-in_min)*(out_max-out_min)/(in_max-in_min)+out_min
def normalize(x):
    in_min=min(x)
    in_max=max(x)
    print(in_min,in_max,"min max")
    out_min=0
    out_max=1
    x_p=[normalize_one(elt,in_min,in_max,out_min,out_max) for elt in x]
    return x_p

def compute_area(l):
    # Calculate the area under the curve using the trapezoidal rule
    x=[elt[0] for elt in l]
    y=[elt[1] for elt in l]
    s=sum([elt[2]**2 for elt in l])  
    x=normalize(x)
    area = np.trapz(y, x)
    return area,s

def sort_curves_by_dataset(curves):
    curves_by_dataset=defaultdict(dict)
    for k,v in curves.items():
        curves_by_dataset[k[1]][k[0]]=v
    
    return curves_by_dataset

def find_x_max(curves):
    threshold=0.95
    x_max=10000
    for k,v in curves.items():
        
        for (cluster_nb,coverage,std) in v:
            if coverage>0.95 and cluster_nb<x_max:
                x_max=cluster_nb
    return x_max

def truncate_curves(curves,x_max):
    new_curves={}
    for k,v in curves.items():
        curve=[]
        for (cluster_nb,coverage,std) in v:
            if cluster_nb<=x_max:
                curve.append((cluster_nb,coverage,std))
        new_curves[k]=curve
    return new_curves

def get_area(filename):
    data = []
    with open(filename, 'r') as file:
        for line in file:
            data.append(json.loads(line))

    curves=defaultdict(list)
    for entry in data:
        name_exp = entry['name_exp']
        dataset_name = entry['dataset_name']
        cluster_nb = entry['cluster_nb']
        pattern_coverage = entry['pattern_coverage'][0] 
        std=entry['pattern_coverage'][0]

        key = (name_exp, dataset_name)
        
        curves[key].append((cluster_nb,pattern_coverage,std))

    for k,v in curves.items():
        curves[k]=sorted(v, key=lambda tup: tup[0])

    curves_by_dataset=sort_curves_by_dataset(curves)
    areas=defaultdict(dict)
    for dataset_name,curves_dict in curves_by_dataset.items():
        x_max=find_x_max(curves_dict)
        curves=truncate_curves(curves_dict,x_max+1)
        for name_exp, value in curves.items():
            print(name_exp)
            a,s=compute_area(value)
            areas[dataset_name][name_exp]=(a,s)

    min_cluster_nb=defaultdict(dict)
    for dataset_name,curves_dict in curves_by_dataset.items():
        
        for name_exp, value in curves_dict.items():
            min_cluster_80=inf
            min_cluster_90=inf
            for cluster_nb,coverage,std in value:
                if coverage>0.80 and cluster_nb<min_cluster_80:
                    min_cluster_80=cluster_nb
                if coverage>0.90 and cluster_nb<min_cluster_90:
                    min_cluster_90=cluster_nb
            min_cluster_nb[dataset_name][name_exp]=(min_cluster_80,min_cluster_90)
    
    sum_areas=defaultdict(int)
    sum_std=defaultdict(int)
    for dataset_name,name_exp_results in areas.items():
        for name_exp, (area,std) in name_exp_results.items():
            sum_areas[name_exp]+=area/4
            
            sum_std[name_exp]+=std
    
    for name_exp,std in sum_std.items():
        print("std",std,sqrt(std ))
        sum_std[name_exp]=sqrt(std)

    areas2=defaultdict(dict)
    for dataset_name,name_exp_results in areas.items():
        for name_exp, (area,std) in name_exp_results.items():
            areas2[dataset_name][name_exp]=area
    sum_areas_list=list(sum_areas.items())

    results_details={}
    for dataset_name,name_exp_results in areas2.items():
        results_details[dataset_name]=sorted(list(name_exp_results.items()), key=lambda tup: tup[1],reverse=True)

    return sorted(sum_areas_list, key=lambda tup: tup[1],reverse=True),results_details,min_cluster_nb,sum_std





if __name__=='__main__':
    filters={"femto_booking_agilkia_v6":0.01,"scanette_100043-steps":0.05,"spree_5000_session_wo_responses_agilkia":0.5,"teaming_execution":0.001}
    results_path='./results/jsonl/all_data_26092.jsonl'

    global_areas,details,min_cluster_nb,sum_std=get_area(results_path)
    for i,elt in enumerate(global_areas):
        print(elt[0].replace('|','&')+'&'+str(round(elt[1],4))+'&'+'\\\\')
    print('----\n')



    

    it=defaultdict(list)
    for dataset_name,areas in details.items():
        print(dataset_name+'----\n')
        list_items=[]
        for area in areas:
            # print(area)
            it[area[0].replace('|','&')]+=[str(area[1]),str(min_cluster_nb[dataset_name][area[0]][0]),str(min_cluster_nb[dataset_name][area[0]][1])]



    

    def custom_sort_key(key):
        parts = key.split('&')
        preprocessing = parts[0]
        clustering = parts[1]
        method = parts[2]
        t=0
        # Sort by method first (BUC before RS)
        if method == "BUC":
            t+=0
        else: 
            t+=2

        if '-' in preprocessing:
            t+=1

        return (t, preprocessing)




        # Sort the dictionary keys using the custom key function
    sorted_keys = sorted(it.keys(), key=custom_sort_key)

    # Create a new dictionary with the sorted keys
    sorted_dict = {key: it[key] for key in sorted_keys}

    # Print the sorted dictionary
    
    it=sorted_dict 



    def transfo(a,bold=False):
        if '0.' in str(a):
            a=str(round(float(a),3))
        if str(a)=='inf':
            a='$\infty$'
        if bold:
            return '\multicolumn{1}{c|}{\cellcolor{tabcolor}{\\textbf{'+a+'}}}'
        else:
            return '\multicolumn{1}{c|}{'+a+'}'
    


    def multi(s):
        return '\multicolumn{1}{|l|}{'+s+'}'

    tab=[]

    lim=34

    for dataset_name,areas in details.items():
        print(dataset_name+'----\n')
        list_items=[]
        for area in areas[:lim]:
                pipeline_str='&'.join([multi(elt) for elt in area[0].split('|')])
                cauc_str=str(round(area[1],3))
                min_c_80_str=str(min_cluster_nb[dataset_name][area[0]][0])
                min_c_90_str=str(min_cluster_nb[dataset_name][area[0]][1])
                
                s=pipeline_str+'&'+cauc_str
                tab.append(s)
                print(len(tab))

    o=""
    l=len(list(details['scanette_100043-steps'])[:lim])
    for i,key in enumerate(list(details['scanette_100043-steps'][:lim])):
        print(len(tab))
        print("lol")
        o+=str(i+1)+'&'+tab[i+0*l]+'&'+tab[i+l]+'&'+tab[i+2*l]+'&'+tab[i+3*l]+'\\\\ \hline \n'
    
    print(o)

