class Config:
    def __init__(self) -> None:
        self.batchsize = 1
        self.datapath = '/home/zhangkechi/workspace/treecaps-master/codenet_data/Project_CodeNet_Java250_spts/'
        self.n_classes = 250
        self.cuda = "1"
        self.modelpath = "model/Java250_8_8_concat"
        self.lr = 0.001

        self.embedding_lookup_lens = (115, 107)

myconfig = Config()

# tail -f java87_nobatch.log
