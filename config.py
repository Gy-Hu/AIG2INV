
class ConfigClassTest:
    def __init__(self):
        self.lr = 1e-3
        self.weight_decay = 1e-10
        self.grad_clip = 0.65
        self.epochs = 150
        self.pretrain_epoch=30

        self.device='cuda:1'

        self.dataset='/data/hongcezh/clause-learning/data-collect/data2dataset/newgraph/test.pkl'
        self.batch_size=5
        self.use_size_below_this = 1000 # if set to 0 then use all graphs
        self.clause_clip = 0
        self.alpha = 0.1 # 0.1  # alpha * (1 - max_per_clause( logit**2 ) )
        self.autoweight=True
        self.var_coeff=-0.001
        self.auto_var_coeff = True
        self.pretrain_iter=5

        self.seed = 1000

        self.nvt=7
        self.vhs = 100
        self.nrounds = 15

        self.save_interval=50
        self.continue_from_model=None
        self.grad_clip=0.65
        self.modelname='test_v1'

    def to_str(self, printfn):
      allvars = vars(self)
      for n,v in allvars.items():
        if len(n)>2 and n[0] != '_' and n[-1] != '_':
          printfn(n + ' := ' + str(v))


class ConfigClassTestZero:
    def __init__(self):
        self.lr = 1e-2
        self.weight_decay = 1e-10
        self.grad_clip = 0.65
        self.epochs = 150
        self.pretrain_epoch=30

        self.device='cuda:1'

        self.dataset=['/data/hongcezh/clause-learning/data-collect/data2dataset/newgraph/test3zeros.pkl', '/data/hongcezh/clause-learning/data-collect/data2dataset/newgraph/test.pkl']
        self.batch_size=5
        self.use_size_below_this = 1000 # if set to 0 then use all graphs
        self.clause_clip = 0
        self.alpha = 0.1 # 0.1  # alpha * (1 - max_per_clause( logit**2 ) )
        self.autoweight=False # self.autoweight=True
        self.var_coeff=-0.1 # self.var_coeff=-0.001
        self.auto_var_coeff = True
        self.pretrain_iter=5

        self.seed = 1000

        self.nvt=7
        self.vhs = 100
        self.nrounds = 15

        self.save_interval=50
        self.continue_from_model=None
        self.grad_clip=0.65
        self.modelname='test_v2zeros'

    def to_str(self, printfn):
      allvars = vars(self)
      for n,v in allvars.items():
        if len(n)>2 and n[0] != '_' and n[-1] != '_':
          printfn(n + ' := ' + str(v))




class ConfigClassHWMCC07_1000:
    def __init__(self):
        self.lr = 1e-2
        self.weight_decay = 1e-10
        self.grad_clip = 0.65
        self.epochs = 150
        self.pretrain_epoch=30

        self.device='cuda:1'

        self.dataset='/data/hongcezh/clause-learning/data-collect/data2dataset/newgraph/hwmcc07dataset.pkl'
        self.batch_size=5
        self.use_size_below_this = 1000 # if set to 0 then use all graphs
        self.clause_clip = 500
        self.alpha = 0.1  # alpha * (1 - max_per_clause( logit**2 ) )
        self.autoweight=True
        self.var_coeff=-0.001
        self.auto_var_coeff = True
        self.pretrain_iter=5

        self.seed = 1000

        self.nvt=7
        self.vhs = 100
        self.nrounds = 15

        self.save_interval=50
        self.continue_from_model=None
        self.grad_clip=0.65
        self.modelname='HWMCC07_1000_v1'

    def to_str(self, printfn):
      allvars = vars(self)
      for n,v in allvars.items():
        if len(n)>2 and n[0] != '_' and n[-1] != '_':
          printfn(n + ' := ' + str(v))


class ConfigClassHWMCC_ALL_under5000node:
    def __init__(self):
        self.lr = 1e-3
        self.weight_decay = 1e-10
        self.grad_clip = 0.65
        self.epochs = 150
        self.pretrain_epoch=35
        self.autoweight=True
        self.var_coeff=-0.001
        self.auto_var_coeff = False
        self.pretrain_iter=5

        self.device='cuda:1'

        self.dataset='/data/hongcezh/clause-learning/data-collect/data2dataset/newgraph/hwmcc07_10_17_20_5000node.pkl'
        self.batch_size=5
        self.use_size_below_this = 5000 # if set to 0 then use all graphs
        self.clause_clip = 500
        self.alpha = 0.1  # alpha * (1 - max_per_clause( logit**2 ) )

        self.seed = 1000

        self.nvt=7
        self.vhs = 100
        self.nrounds = 15

        self.save_interval=50
        self.continue_from_model=None
        self.grad_clip=0.65
        self.modelname='HWMCC_ALL_under5000node_v1'

    def to_str(self, printfn):
      allvars = vars(self)
      for n,v in allvars.items():
        if len(n)>2 and n[0] != '_' and n[-1] != '_':
          printfn(n + ' := ' + str(v))

config = ConfigClassHWMCC_ALL_under5000node()
# config = ConfigClassTestZero()
# config = ConfigClassTest()


