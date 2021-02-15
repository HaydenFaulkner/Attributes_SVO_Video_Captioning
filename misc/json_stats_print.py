
import json
import os

def main(experiment, dataset, key='TrainLoss'):
    json_file = os.path.join('../experiments', experiment, dataset + '_history.json')
    with open(json_file) as f:
        data = json.load(f)

    for i in range(200):
        if str(i) in data.keys():
            print(data[str(i)][key])
    print()

# main('transformer02_111111_ksvo', 'msvd')#, key='CIDEr')
# main('transformer02_111111_ksvo', 'msrvtt', key='CIDEr')
# main('transformer02_111111_ksvo', 'msrvtt', key='Train_CIDEr')
# main('transformer00', 'msvd', key='CIDEr')
# main('transformer00_2_1_2048', 'msvd', key='TrainLoss')
# main('transformer00_2_1_2048', 'msvd', key='Train_CIDEr')
# main('transformer00_2_1_2048', 'msvd', key='CIDEr')
main('transformer00_1_8_2048', 'msrvtt', key='TrainLoss')