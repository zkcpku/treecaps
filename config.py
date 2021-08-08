class Config:
    def __init__(self) -> None:
        self.batchsize = 8
        self.datapath = '/home/zhangkechi/workspace/treecaps-master/codenet_data/Project_CodeNet_Java250_spts/'
        self.n_classes = 250
        self.cuda = "1"
        self.modelpath = "model/Java250_8_7_batch"
        self.lr = 0.001 * 0.1

myconfig = Config()

# tail -f java87_nobatch.log
