from lib.generator_dependencies import *
from lib.general_dependencies import *
from utils.general import *

class CustomCallback(keras.callbacks.Callback):
    """ A class to create custom callback options. This overrides a set of methods called at various stages of training, testing, and predicting. 
        Callbacks are useful to get a view on internal states and statistics of the model during training.
            Callback list can be passed for .fit(), .evaluate(), and .predict() methods
            
        keys = list(logs.keys())
    """
    def __init__(self, out_file=sys.stdout):
        super(CustomCallback, self).__init__()
        self.start_time = None
        self.out_file = out_file
        
    def on_train_begin(self, logs=None):
        self.start_time = time.time()
        write("Training Beginning:\t "+ time.ctime(self.start_time), self.out_file)

    def on_train_end(self, logs=None):
        write("Training Complete --- %s hours ---" % ((time.time() - self.start_time)/3600), self.out_file)

    def on_epoch_begin(self, epoch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        pass

    def on_test_begin(self, logs=None):
        self.start_time = time.time()
        write("Evaluation Beginning:\t "+ time.ctime(self.start_time), self.out_file)

    def on_test_end(self, logs=None):
        write("Evaluation Complete --- %s minutes ---" % ((time.time() - self.start_time)/60), self.out_file)

    def on_predict_begin(self, logs=None):
        self.start_time = time.time()
        write("Predicting Beginning:\t "+ time.ctime(self.start_time), self.out_file)

    def on_predict_end(self, logs=None):
        write("Predicting Complete --- %s minutes ---" % ((time.time() - self.start_time)/60), self.out_file)

    def on_train_batch_begin(self, batch, logs=None):
        pass

    def on_train_batch_end(self, batch, logs=None):
        pass

    def on_test_batch_begin(self, batch, logs=None):
        pass

    def on_test_batch_end(self, batch, logs=None):
        pass

    def on_predict_batch_begin(self, batch, logs=None):
        pass

    def on_predict_batch_end(self, batch, logs=None):
        pass 