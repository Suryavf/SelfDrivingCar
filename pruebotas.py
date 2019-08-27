from ImitationLearning.ImitationModel import ImitationModel
import config


init    = config.Init()
setting = config.Setting()

test = ImitationModel(init,setting)
test.build()
test.execute()
