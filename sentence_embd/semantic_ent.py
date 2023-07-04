import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer, util
import sys

MODEL_NAME = 'all-MiniLM-L6-v2'
model = SentenceTransformer(MODEL_NAME)
model.max_seq_length = 512

def read_json(file_path):
    
    with open(file_path, "r") as json_file:
        data = json.load(json_file)
    system_messages = [dict["response"] for dict in data["system"]]
    return system_messages

def plot_scores(data_set):
    data = np.triu(data_set, k=0)
    colormap = sns.color_palette("mako", as_cmap=True)
    ax = sns.heatmap(data, linewidths=0.5, cmap=colormap, annot=True)
    plt.title = ('Semantic Entropy Across Epochs')
    plt.show()

def sequential_score(sys_messages):
    sys_messages1 = ["Start"] + sys_messages
    sys_messages2 = sys_messages + ["End"]
    #Compute embedding for both lists
    embeddings1 = model.encode(sys_messages1, convert_to_tensor=True)
    embeddings2 = model.encode(sys_messages2, convert_to_tensor=True)
    #Compute cosine-similarities
    cosine_scores = util.cos_sim(embeddings1, embeddings2)
    scores = [cosine_scores[i][i] for i in range(1, len(sys_messages1)-1)]

    for k, item in enumerate(scores):
        print("Entropy Score between Epoch {} and Epoch {}: {:.4f}".format(k + 1, k + 2, item))


def point_to_point_score(sentences, n_epoch_1, n_epoch_2):

    #Compute embedding for both lists
    embeddings = model.encode(sentences, convert_to_tensor=True)

    #Compute cosine-similarities
    cosine_scores = util.cos_sim(embeddings, embeddings)
    
    score = cosine_scores[n_epoch_1][n_epoch_2]

    print("Entropy Score between Epoch {} and Epoch {}: {:.4f}".format(n_epoch_1 + 1, n_epoch_2 + 1, score))

def all_scores(sys_messages):

    #Compute embedding for both lists
    embeddings = model.encode(sys_messages, convert_to_tensor=True)

    #Compute cosine-similarities
    cosine_scores = util.cos_sim(embeddings, embeddings)

    for i in range(cosine_scores.size()[0]):
        for j in range(i + 1):
            if i == j: continue
            score = cosine_scores[i][j]
            print("Entropy Score between Epoch {} and Epoch {}: {:.4f}".format(j + 1, i + 1, score))

    return cosine_scores

def main():
    arguments = sys.argv
    sys_messages = read_json(arguments[1])

    if len(arguments) < 2:
        raise Exception("No file was provided")
    else:
        arg3 = arguments[2]
        if arg3 == "-p":
            sequential_score(sys_messages)
        elif arg3 == "-a":
            scores = all_scores(sys_messages)
            plot_scores(scores)
        else:
            point_to_point_score(sys_messages, int(arguments[2]), int(arguments[3]))

if __name__ == '__main__':
    main()