
class ConfigClass:
    def __init__(self):
        self.lr = 1e-2
        self.weight_decay = 1e-10
        self.grad_clip = 0.65
        self.epochs = 150

        self.device='cuda:0'

        self.dataset='/data/hongcezh/clause-learning/data-collect/data2dataset/test.pkl'
        self.batch_size=5
        self.use_size_below_this = 1000 # if set to 0 then use all graphs

        self.seed = 1000

        self.nvt=6
        self.vhs = 100
        self.nrounds = 15

        self.save_interval=50
        self.continue_from_model=None
        self.grad_clip=0.65

    def to_str(self, printfn):
      allvars = vars(self)
      for n,v in allvars.items():
        if len(n)>2 and n[0] != '_' and n[-1] != '_':
          printfn(n + ' := ' + str(v))

config = ConfigClass()
