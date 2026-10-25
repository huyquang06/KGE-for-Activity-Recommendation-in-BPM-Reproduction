import numpy as np

#context_file = "/work/dsola/kge/local/last_rev_forAllCaise_context/test_context.txt"
#prediction_file = "/work/dsola/kge/local/AnyBURL_files/test-predictions-c5-a2-600-inProcess.txt"
#RLRec=False
#filtered_prediction_file = prediction_file.replace(".txt","_filtered.txt")


def main_eval(context_file,prediction_file,RLRec,filtered_prediction_file):
    contexts = dict()
    read_context(context_file,contexts,RLRec)

    hitAtRates, mrr, hitAtRatesBefore, mrrBefore = filter_and_evaluate(prediction_file,filtered_prediction_file,contexts,RLRec)

    with open(filtered_prediction_file,"a") as fpf:
        for i in range(len(hitAtRates)):
            if i == 10:
                string = "all"
            else:
                string = str(i+1)
            print("Hits at " + string + " Rate: " + str(hitAtRates[i]))
            print("Hits at " + string + " Rate Before Filtering: " + str(hitAtRatesBefore[i]))
            fpf.write("Hits at " + string + " Rate: " + str(hitAtRates[i]) + "\n")
            fpf.write("Hits at " + string + " Rate Before Filtering: " + str(hitAtRatesBefore[i]) + "\n")
        print("Hits at 10 improvement through filtering: " + str(hitAtRates[9]-hitAtRatesBefore[9]))
        fpf.write("Hits at 10 improvement through filtering: " + str(hitAtRates[9]-hitAtRatesBefore[9])+"\n")
        print("MRR: " + str(mrr))
        print("MRR Before: " + str(mrrBefore))
        fpf.write("MRR: " + str(mrr)+"\n")
        fpf.write("MRR Before: " + str(mrrBefore)+"\n")
        print("MRR improvement through filtering: " + str(mrr-mrrBefore))
        fpf.write("MRR improvement through filtering: " + str(mrr-mrrBefore)+"\n")


# read in the context file

def read_context(context_file,contexts,RLRec):
    with open(context_file,"r") as cf:
        pro=0
        for l in cf:
            pro+=1
            lsplit = l.split("\t")[:-1] # last element is empty
            for t in lsplit[1:]:
                if len(t.split(",")) == 1: # label atom
                    label = t.split("(")[0]
                    node = t.split("(")[1].replace(")","")
                    if RLRec:
                        process = pro
                    else:
                        process = "_".join(node.split("_")[1:])
                    if process in contexts:
                        contexts[process]["labels"].add(label)
                    else:
                        contexts[process] = {"labels": set([label])}

# read the prediction file, filter and calculate the metrics

def filter_and_evaluate(prediction_file,filtered_prediction_file,contexts,RLRec):
    numberOfPredictions = 0
    hitsAt = np.zeros(11) # Recall = hitsAtTen/numberOfPredictions
    hitsAtBefore = np.zeros(11)
    mrrNumerator = 0 # MRR = mrrNumerator/numberOfPredictions
    mrrNumeratorBefore = 0
    with open(prediction_file,"r") as pf:
        with open(filtered_prediction_file,"w") as fpf:
            correctLabel = ""
            process = ""
            pro = 0
            for l in pf:
                if "Heads" in l:
                    continue
                elif "Tails" in l:
                    lnew = l.split("Tails: ")[1].split("\t")
                    predictions = lnew[::2]
                    filtered_predictions, sortedOutNodeOrProcess = filter_predictions(predictions,contexts,process)
                    fpf.write("Tails: " + "\t".join(filtered_predictions) + "\n")
                    hitsAtCase,mrrNumeratorCase = evaluate(correctLabel,filtered_predictions)
                    hitsAtCaseBefore,mrrNumeratorCaseBefore = evaluate(correctLabel,predictions)
                    hitsAt += hitsAtCase
                    hitsAtBefore += hitsAtCaseBefore
                    mrrNumerator += mrrNumeratorCase
                    mrrNumeratorBefore += mrrNumeratorCaseBefore
                    #if mrrNumerator > mrrNumeratorBefore and sortedOutNodeOrProcess:
                    if hitsAtCase[6] > hitsAtCaseBefore[6] and sortedOutNodeOrProcess:
                        print(process)
                else:
                    pro+=1
                    fpf.write(l)
                    if RLRec == True:
                        correctLabel = l.split("after ")[1].replace("\n","")
                        process = pro
                    else:
                        correctLabel = l.split("hasLabel ")[1].replace("\n","")
                        node = l.split(" hasLabel ")[0]
                        process = "_".join(node.split("_")[1:])
                    numberOfPredictions += 1
    print(numberOfPredictions)
    return hitsAt/numberOfPredictions, mrrNumerator/numberOfPredictions, hitsAtBefore/numberOfPredictions, mrrNumeratorBefore/numberOfPredictions

def filter_predictions(predictions,contexts,process):
    filtered_predictions = []
    sortedOutNodeOrProcess = False
    for idx,p in enumerate(predictions):
        if p in contexts[process]["labels"]:
            continue
        if "_" in p: # node or process
            if idx<10:
                sortedOutNodeOrProcess = True
            continue
        filtered_predictions.append(p)
    return filtered_predictions, sortedOutNodeOrProcess


def evaluate(correctLabel,filtered_predictions):
    hitsAt = np.zeros(11)
    mrrNumerator = 0
    for i in range(len(filtered_predictions[:10])):
        if filtered_predictions[i] == correctLabel:
            for j in range(i,10):
                hitsAt[j] += 1
            mrrNumerator += 1/(i+1)
            break
    if correctLabel in filtered_predictions:
        hitsAt[10] += 1
    return hitsAt, mrrNumerator

#main_eval(context_file,prediction_file,RLRec,filtered_prediction_file)