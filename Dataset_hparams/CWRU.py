class CWRU():
    def __init__(self):
        super().__init__()
        self.train_params = {
            'N_epochs': 100,
            'batch_size': 32,
            'weight_decay': 1e-4,
            'step_size': 50,
            'lr_decay': 0.5
        }
        self.alg_hparams = {
            'NO_ADAPT': {'learning_rate': 1e-3, 'src_cls_loss_wt': 1},
            'TARGET_ONLY': {'learning_rate': 1e-3, 'trg_cls_loss_wt': 1},

            "SASA": {
                "domain_loss_wt": 5.760124609738364,
                "learning_rate": 0.001,
                "src_cls_loss_wt": 4.130742585941761,
                "weight_decay": 0.0001
            },
            "DSAN": {
                "learning_rate": 0.0005,
                "mmd_wt": 0.5993593617252002,
                "src_cls_loss_wt": 0.386167577207679,
                "domain_loss_wt": 0.16,
                "weight_decay": 0.0001
            },
            "CoDATS": {
                "domain_loss_wt": 9.314114040099962,
                "learning_rate": 0.0005,
                "src_cls_loss_wt": 7.700018679383289,
                "weight_decay": 0.0001
            },
            "HoMM": {
                "hommd_wt": 7.172430927893522,
                "learning_rate": 0.0005,
                "src_cls_loss_wt": 0.20121211752349172,
                "domain_loss_wt": 0.9824,
                "weight_decay": 0.0001
            },
            "DIRT": {
                "cond_ent_wt": 1.329734510542011,
                "domain_loss_wt": 6.632293308809388,
                "learning_rate": 0.001,
                "src_cls_loss_wt": 7.729881324550688,
                "vat_loss_wt": 6.912258476982827,
                "weight_decay": 0.0001
            },
            "AdvSKM": {
                "domain_loss_wt": 1.8649335076712072,
                "learning_rate": 0.001,
                "src_cls_loss_wt": 3.961611563054495,
                "weight_decay": 0.0001
            },
            "DDC": {
                "learning_rate": 0.0005,
                "mmd_wt": 8.355791702302787,
                "src_cls_loss_wt": 1.2079058664226126,
                "domain_loss_wt": 0.2048,
                "weight_decay": 0.0001
            },
            "CDAN": {
                "cond_ent_wt": 0.1841898900507932,
                "domain_loss_wt": 1.9307294194382076,
                "learning_rate": 0.0005,
                "src_cls_loss_wt": 4.15410157776963,
                "weight_decay": 0.0001
            },
            "DANN": {
                "domain_loss_wt": 1.0296390274908802,
                "learning_rate": 0.0005,
                "src_cls_loss_wt": 2.038458138479581,
                "weight_decay": 0.0001
            },
            "Deep_Coral": {
                "coral_wt": 5.9357031653707475,
                "learning_rate": 0.0005,
                "src_cls_loss_wt": 0.43859323168654,
                "weight_decay": 0.0001
            },
            "MMDA": {
                "cond_ent_wt": 6.707871745810609,
                "coral_wt": 5.903714930042433,
                "learning_rate": 0.005,
                "mmd_wt": 6.480169289397163,
                "src_cls_loss_wt": 0.18878476669902317,
                "weight_decay": 0.0001
            },
            'CoTMix': {'learning_rate': 0.001, 'mix_ratio': 0.52, 'temporal_shift': 14,
                       'src_cls_weight': 0.8, 'src_supCon_weight': 0.1, 'trg_cont_weight': 0.1,
                       'trg_entropy_weight': 0.05}

        }