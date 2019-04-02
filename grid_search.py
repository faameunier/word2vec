from skipGram import SkipGram
import os
import pandas as pd
import numpy as np
import utils
import logging


def test_model(sg):
    res = []
    for filename in os.listdir("data/"):
        pairs = utils.loadPairs("data/" + filename)
        output = []
        objective = []
        for a, b, o in pairs:
            objective.append(o)
            # print(sg.similarity(a, b))
            output.append(sg.similarity(a, b))
        df = pd.DataFrame(np.transpose(np.array([objective, output])), columns=["objective", "output"])
        res.append(df.corr().values[0][1])

    # print(res)
    print("Average correlation :", np.mean(res))
    print("Best :", np.max(res))
    print("Worst :", np.min(res))


# Estimated time to run ~5h ? on 30k lines 81models
SIZE_EMBEDS = [100, 200, 300]
SIZE_C = [5, 8, 10]
SIZE_NEG = [5, 10, 15]
SUBSAMPLE = [0.001, 0.0001, 0.00001]
DEVSET_PATH = "100klines"

if __name__ == '__main__':
    logging.basicConfig(filename='info.log', level=logging.INFO)
    sentences = utils.text2sentences(DEVSET_PATH)
    for emb in SIZE_EMBEDS:
        for c in SIZE_C:
            for neg in SIZE_NEG:
                for sample in SUBSAMPLE:
                    print("=============================")
                    print("------------START------------")
                    print("nEmbed", emb)
                    print("negativeRate", neg)
                    print("winSize", c)
                    print("sampling_rate", sample)
                    sg = SkipGram(sentences, nEmbed=emb, negativeRate=neg, winSize=c, minCount=5, sampling_rate=sample)
                    sg.init_embeddings()
                    sg.train(epochs=3, n_worker=8)
                    test_model(sg)
                    print("-------------END-------------")
