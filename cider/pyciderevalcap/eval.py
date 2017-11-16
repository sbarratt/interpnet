__author__ = 'rama'
from .tokenizer.ptbtokenizer import PTBTokenizer
from .cider.cider import Cider
from .ciderD.ciderD import CiderD


class CIDErEvalCap:
    def __init__(self, gts, res, df):
        tokenizer = PTBTokenizer('gts')
        _gts = tokenizer.tokenize(gts)
        tokenizer = PTBTokenizer('res')
        _res = tokenizer.tokenize(res)

        self.gts = _gts
        self.res = _res
        self.df = df

    def evaluate(self):
        # =================================================
        # Set up scorers
        # =================================================

        scorers = [
            (Cider(df=self.df), "CIDEr"), (CiderD(df=self.df), "CIDErD")
        ]

        # =================================================
        # Compute scores
        # =================================================
        metric_scores = {}
        for scorer, method in scorers:
            print ('computing %s score...' % (scorer.method()))
            score, scores = scorer.compute_score(self.gts, self.res)
            print ("Mean %s score: %0.3f" % (method, score))
            metric_scores[method] = list(scores)
        return metric_scores
