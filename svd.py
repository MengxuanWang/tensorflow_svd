import numpy as np
from utils import *

class SvdMatrix:

    '''
    trainfile -> name of file of train data against
    r -> rank of approximates (for U and V)
    steps -> train steps
    lrate -> learning rate
    regularizer -> regularizer
    '''
    def __init__(self, trainfile, nsample=0, r=30, steps=25, lrate=0.1, regular=0.005):
        self.trainrats, self.nuser, self.nitem, self.mu = load_ratings(trainfile, nsample)
        print "user number : ", self.nuser
        print "item number : ", self.nitem
        print "average rating : ", self.mu

        self.r = r
        self.lrate = lrate
        self.regular = regular
        self.steps = steps

        self.__init_bias_lfm()

    def __init_bias_lfm(self):
        self.bu = [0] * (self.nuser)
        self.bv = [0] * (self.nitem)
        self.U = [[random.random()/np.sqrt(self.r) for _ in range(self.r)] for idx in xrange(self.nuser+1)]
        self.V = [[random.random()/np.sqrt(self.r) for _ in range(self.r)] for idx in xrange(self.nitem+1)]

    def train(self):
        print "lrate={}, regular={}, r={}, steps={}".format(self.lrate, self.regular, self.r, self.steps)

        for step in range(self.steps):
            print "step: ", step
            for u, v, r, t in self.trainrats:
                ri = self.predict(u, v)
                err = r - ri

                uTemp = self.bu[u]
                vTemp = self.bv[v]
                self.bu[u] += self.lrate * (err - self.regular * uTemp)
                self.bv[v] += self.lrate * (err - self.regular * vTemp)

                for k in range(self.r):
                    uTemp = self.U[u][k]
                    vTemp = self.V[v][k]
                    self.U[u][k] += self.lrate * (err * vTemp - self.regular * uTemp)
                    self.V[v][k] += self.lrate * (err * uTemp - self.regular * vTemp)
            rmse = self.calcu_rmse(self.trainrats)
            print "rmse: ", rmse
            self.lrate = self.lrate * 0.9

    def calcu_rmse(self, rats):
        sse = .0
        for u, v, r, _ in rats:
            sse += (r - self.predict(u, v))**2
        return np.sqrt(sse / len(rats))

    def predict(self, uid, iid):
        assert uid <= self.nuser
        assert iid <= self.nitem

        rat = self.mu
        rat += self.bu[uid] + self.bv[iid]
        rat += dot_product(self.U[uid], self.V[iid])
        if rat > 5:
            rat = 5
        if rat < 1:
            rat = 1
        return rat

def predicts(model, filepath):
    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    respath = os.path.join(out_dir, "res.csv")
    with open(filepath) as rf, \
      open(respath, 'wb') as wf:
        w = csv.writer(wf)
        w.writerow(["score"])
        for line in islice(rf, 1, None):
            u, v = [int(it) for it in line.strip().split(",")]
            if u > model.nuser or v > model.nitem:
                w.writerow([model.mu])
                continue
            ri = model.predict(u, v)
            w.writerow([ri])

if __name__ == "__main__":
    svd = SvdMatrix("train.csv")
    svd.train()
    predicts(svd, "test.csv")
