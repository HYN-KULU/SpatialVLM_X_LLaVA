import json
import torch
filename = './test.jsonl'
dataset=[]
with open(filename, 'r') as file:
    for line in file:
        data = json.loads(line.strip())
        dataset.append(data)
correct=0
correct_probability=0
pro=torch.load("./save_llava_finetune_vsr.pth")
bias_pro_list=torch.load("./save_llava_finetune_vsr_bias.pth")
relations={}
for i in range(60):
    pred_pro=(pro[i]["Yes"]/(pro[i]["Yes"]+pro[i]["No"]),pro[i]["No"]/(pro[i]["Yes"]+pro[i]["No"]))
    bias_pro=(bias_pro_list[i]["Yes"]/(bias_pro_list[i]["Yes"]+bias_pro_list[i]["No"]),bias_pro_list[i]["No"]/(bias_pro_list[i]["Yes"]+bias_pro_list[i]["No"]))
    weighted_yes=pred_pro[0]/bias_pro[0]
    weighted_no=pred_pro[1]/bias_pro[1]
    weighted=(weighted_yes/(weighted_yes+weighted_no),weighted_no/(weighted_yes+weighted_no))
    label=dataset[i]['label']
    relation=dataset[i]['relation']
    if relation not in relations.keys():
        relations[relation]={"correct":[],"prob":[]}
    weighted_answer="Yes" if weighted[0]>weighted[1] else "No"
    if (label==1 and weighted_answer=="Yes") or (label==0 and weighted_answer=="No"):
        correct+=1
        relations[relation]["correct"].append(1)
        print(i," right")
    else:
        relations[relation]["correct"].append(0)
        print(i," wrong")
    if (label==1):
        correct_probability+=weighted[0]
        relations[relation]["prob"].append(weighted[0].item())
    else:
        correct_probability+=weighted[1]
        relations[relation]["prob"].append(weighted[1].item())
for relation in relations:
    relations[relation]["correct"]=sum(relations[relation]["correct"])/len(relations[relation]["correct"])
    relations[relation]["prob"]=sum(relations[relation]["prob"])/len(relations[relation]["prob"])
print("Testint Accuracy Performance: ",correct/60)
print("Testint Accuracy Probability Performance: ",(correct_probability/60).item())
print(relations)
