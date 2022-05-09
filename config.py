class BaseConfiguration:
    def __init__(self):
        self.lr = 1e-3
        self.weight_decay = 1e-10
        self.grad_clip = 0.65
        self.epochs = 170
        self.auto_pretrain = True

        self.device='cuda:1'

        self.batch_size=5
        self.use_size_below_this = 1000 # if set to 0 then use all graphs
        self.clause_clip = 0
        self.alpha = 0.1 # 0.1  # alpha * (1 - max_per_clause( logit**2 ) )
        self.autoweight=False
        self.var_coeff=0
        self.auto_var_coeff = False

        self.seed = 1000

        self.nvt=7
        self.vhs = 100
        self.nrounds = 15

        self.continue_from_model=None

    def to_str(self, printfn):
      allvars = vars(self)
      for n,v in allvars.items():
        if len(n)>2 and n[0] != '_' and n[-1] != '_':
          printfn(n + ' := ' + str(v))

class ConfigClassTest(BaseConfiguration):
    def __init__(self):
        super().__init__()
        self.dataset='/data/hongcezh/clause-learning/data-collect/data2dataset/newgraph/test.pkl'
        self.modelname='test_v1'

class ConfigClassTestZero(BaseConfiguration):
    def __init__(self):
        super().__init__()
        self.auto_pretrain = True
        self.dataset=['/data/hongcezh/clause-learning/data-collect/data2dataset/newgraph/test3zeros.pkl', '/data/hongcezh/clause-learning/data-collect/data2dataset/newgraph/test.pkl']
        self.modelname='test_v2zeros'


class ConfigClassHWMCC07_1000(BaseConfiguration):
    def __init__(self):
        super().__init__()
        self.dataset='/data/hongcezh/clause-learning/data-collect/data2dataset/newgraph/hwmcc07dataset.pkl'
        self.modelname='HWMCC07_1000_v1'


class ConfigClassHWMCC_ALL_under5000node(BaseConfiguration):
    def __init__(self):
        super().__init__()
        self.dataset='/data/hongcezh/clause-learning/data-collect/data2dataset/newgraph/hwmcc07_10_17_20_5000node.pkl'
        self.modelname='HWMCC_ALL_under5000node_v1'
        self.use_size_below_this = 1000 # if set to 0 then use all graphs
        self.clause_clip = 200
        # TODO: try: self.autoweight=True
        self.autoweight=True


config = ConfigClassHWMCC_ALL_under5000node()
# config = ConfigClassTestZero()
# config = ConfigClassTest()


