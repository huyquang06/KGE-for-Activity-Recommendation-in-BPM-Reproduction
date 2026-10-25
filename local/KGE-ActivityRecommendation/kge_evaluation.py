# run with -Xutf8

import torch
from kge.model import KgeModel
from kge.util.io import load_checkpoint
import os

from filter_and_evaluate import main_eval

# User defined section
model_ids = ["distmult"] #model_ids = ["RotatE", "distmult", "hitter", "TuckER", "TransE"]
job_type = "train" # train or search  # if search please specify the best trial folder
dataset_name = "bpmai_lastrev_caise_onlyAfter"
result_folder_name = "20231123-122024-distmult-KvsAll-kl"
best_trial_name = "00012"


prediction_file = ""

# delimiter
delim = ' '

topk_pre = 1000
topk = 1000

# Get the path to the directory containing the script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Get the path for the relevant dataset
datapath = os.path.join(script_dir, "data", dataset_name)
# Get the path for the best checkpoint file
if job_type == "train":
    modelpath_kge = os.path.join(script_dir, "local", "experiments", result_folder_name,
                                    "checkpoint_best.pt")
else:
    modelpath_kge = os.path.join(script_dir, "local", "experiments", result_folder_name,
                                     best_trial_name, "checkpoint_best.pt")
# Get the context file path
context_file = os.path.join(script_dir, "local", "KGE-ActivityRecommendation",
                            "evaluation_test_context","test_context.txt")


def main(datapath,modelpath_kge):
    
    prediction_tasks = "test.txt"
    
    for model_id in model_ids:

        outputpath = "/work/kamiriel/kge/local/new results/" + (modelpath_kge.split("experiments/")[1]).split("/")[0]+"-TEST.txt"
        global prediction_file 
        prediction_file = outputpath
        #print(outputpath)

        # entity/number mappings
        e2n = {}
        n2e = {}
        # relation/number mappings
        r2n = {}
        n2r = {}

        # prepare a filter set
        fs = set()

        hits = 0
        num_of_preds = 0

        # read the mappings
        ent_map_path = os.path.join(datapath, "entity_ids.del")
        rel_map_path = os.path.join(datapath, "relation_ids.del")
        read_mapping(ent_map_path, n2e, e2n)
        read_mapping(rel_map_path, n2r, r2n)
        #read_mapping(datapath + "entity_ids.del", n2e, e2n)
        #read_mapping(datapath + "relation_ids.del", n2r, r2n)
        # load the model
        checkpoint = load_checkpoint(modelpath_kge)
        model = KgeModel.create_from(checkpoint)

        prepare_filterset(fs, [])#[datapath + "train.txt", datapath + "valid.txt"])

        out = open(outputpath, "w")
        
        prediction_tasks_path = os.path.join(datapath, prediction_tasks)
        f = open(prediction_tasks_path, "r")
        #f = open(datapath + prediction_tasks, "r")
        counter = 0
        for x in f:
            token = x.split("\t")
            if len(token) == 3:
                s = token[0]; r = token[1]; o = token[2].replace("\n","")
                counter += 1
                print_predictions(fs, e2n, n2e, r2n, n2r, model, s,r,o, out)
            if counter % 100 == 0:
                print("converted", counter, "prediction tasks ...")
        f.close()
        out.close()
        print("created AnyBURL file for ", counter, "triples")
    

def prepare_filterset(fs, pathes):
    for p in pathes:
        f = open(p, "r")
        counter = 0
        for x in f:
            token = x.split("\t")
            if len(token) == 3:
                fs.add(us(token[0], token[1], token[2].replace("\n","")))
                counter += 1
        f.close()
        print("read", counter, "filter triples from", p)
        
def us(t1, t2, t3):
    global delim
    return t1 + delim + t2 + delim + t3

def read_mapping(path, n2x, x2n):
    f = open(path, "r")
    counter = 0
    for x in f:
        token = x.split("\t")
        if len(token) == 2:
            counter += 1
            n2x[int(token[0])] = token[1].replace("\n","")
            x2n[token[1].replace("\n","")] = int(token[0])
    f.close()
    print(">>> read", counter, "mappings from", path,"between ids and numbers")


def print_predictions(fs, e2n, n2e, r2n, n2r, model, s, r, o, out):
    global topk
    global topk_pre
    out.write(s + " " + r + " " + o + "\n")
    tt = us(s,r,o)

    sn = e2n[s]; st = torch.Tensor([sn,]).long()  
    on = e2n[o]; ot = torch.Tensor([on,]).long()   
    rn = r2n[r]; rt = torch.Tensor([rn,]).long() 
     
    # scores for the object/tail
    scores_t = model.score_sp(st, rt)         
    ranking_t = torch.topk(scores_t, topk_pre)
    out.write("Tails: ")
    ti = 0
    for i in range(topk_pre):
        if is_unknown(tt, us(s,r,n2e[ranking_t[1][0][i].item()]), fs):
            #if (tt == us(s,r,n2e[ranking_t[1][0][i].item()])): hits += 1
            ti += 1
            out.write(n2e[ranking_t[1][0][i].item()] + "\t")
            out.write(str(ranking_t[0][0][i].item()) + "\t")
            if ti == topk: break
    out.write("\n")
    if ti < topk: print (">>> filter error tail prediction", ti, tt)
    # scores for the object/tail
    scores_h = model.score_po(rt, ot)         
    ranking_h = torch.topk(scores_h, topk_pre)
    out.write("Heads: ")
    hi = 0
    for i in range(topk_pre):
        if is_unknown(tt, us(n2e[ranking_h[1][0][i].item()],r,o), fs):
            hi += 1
            out.write(n2e[ranking_h[1][0][i].item()] + "\t")
            out.write(str(ranking_h[0][0][i].item()) + "\t")
            if hi == topk: break
    out.write("\n")
    if hi < topk: print (">>> filter error head prediction", hi, tt)


def is_unknown(tt, pred, fs):
    if tt == pred:
        return True
    elif pred in fs:
        return False
    return True

main(datapath,modelpath_kge)
main_eval(context_file,prediction_file,False,prediction_file.replace(".txt","_filtered.txt")) #changed! from True to False

