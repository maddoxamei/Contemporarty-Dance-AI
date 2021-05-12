from lib.dance_classifier_dependencies import *
from lib.general_dependencies import *
from lib.data_dependencies import *
from lib.global_variables import *
from utils import get_unique_dance_names, get_sample_data, write, progressbar, get_save_path


def aggregate_data(out_file=sys.stdout):
    dances = get_unique_dance_names(csv_data_dir)
    comprehensive_train_X = np.array([])
    comprehensive_train_Y = np.array([])
    comprehensive_validate_X = np.array([])
    comprehensive_validate_Y = np.array([])
    comprehensive_evaluation_X = np.array([])
    comprehensive_evaluation_Y = np.array([])
    
    comprehensive_train_Class_Y = np.array([])
    comprehensive_validate_Class_Y = np.array([])
    comprehensive_evaluation_Class_Y = np.array([])
    
    start_time = time.time()
    for dance in progressbar(dances, "Progress: "):
        csv_filename, np_filename = get_save_path(dance)
        train_X, train_Y, validate_X, validate_Y, evaluation_X, evaluation_Y = get_sample_data(csv_filename, np_filename, look_back, offset, forecast, sample_increment, training_split, validation_split, pos_pre_processes, rot_pre_processes)
    
        sentiment = dance.split('_')[-1]
        train_Class_Y = np.full((train_X.shape[0],1),int(sentiment))
        validate_Class_Y = np.full((validate_X.shape[0],1),int(sentiment))
        evaluation_Class_Y = np.full((evaluation_X.shape[0],1),int(sentiment))
        
        if(len(comprehensive_train_X)==0):
            comprehensive_train_X = train_X
            comprehensive_train_Y = train_Y
            comprehensive_validate_X = validate_X
            comprehensive_validate_Y = validate_Y
            comprehensive_evaluation_X = evaluation_X
            comprehensive_evaluation_Y = evaluation_Y
            
            comprehensive_train_Class_Y = train_Class_Y
            comprehensive_validate_Class_Y = validate_Class_Y
            comprehensive_evaluation_Class_Y = evaluation_Class_Y
        else:
            comprehensive_train_X = np.vstack((comprehensive_train_X,train_X))
            comprehensive_train_Y = np.vstack((comprehensive_train_Y,train_Y))
            comprehensive_validate_X = np.vstack((comprehensive_validate_X,validate_X))
            comprehensive_validate_Y = np.vstack((comprehensive_validate_Y,validate_Y))
            comprehensive_evaluation_X = np.vstack((comprehensive_evaluation_X,evaluation_X))
            comprehensive_evaluation_Y = np.vstack((comprehensive_evaluation_Y,evaluation_Y))
            
            comprehensive_train_Class_Y = np.vstack((comprehensive_train_Class_Y,train_Class_Y))
            comprehensive_validate_Class_Y = np.vstack((comprehensive_validate_Class_Y,validate_Class_Y))
            comprehensive_evaluation_Class_Y = np.vstack((comprehensive_evaluation_Class_Y,evaluation_Class_Y))
            
    write("Fetching and Agregating Training Data --- {} seconds ---".format(start_time - time.time()), out_file)  
    
    
    np.save(training_filepath+"_X", comprehensive_train_X)
    np.save(training_filepath+"_Y", comprehensive_train_Y)
    np.save(validation_filepath+"_X", comprehensive_validate_X)
    np.save(validation_filepath+"_Y", comprehensive_validate_Y)
    np.save(evaluation_filepath+"_X", comprehensive_evaluation_X)
    np.save(evaluation_filepath+"_Y", comprehensive_evaluation_Y)
    
    np.save(training_filepath+"_Class_Y", comprehensive_train_Class_Y)
    np.save(validation_filepath+"_Class_Y", comprehensive_validate_Class_Y)
    np.save(evaluation_filepath+"_Class_Y", comprehensive_evaluation_Class_Y)
    
    print("Saved to", training_filepath+"_Class_Y")
    
def get_comprehensive_data():
    if not (os.path.exists(training_filepath+"*.npy")):
        aggregate_data()
    return np.load(training_filepath+"_X.npy"), np.load(training_filepath+"_Y.npy"), np.load(validation_filepath+"_X.npy"), np.load(validation_filepath+"_Y.npy"), np.load(evaluation_filepath+"_X.npy"), np.load(evaluation_filepath+"_Y.npy"),np.load(training_filepath+"_Class_Y.npy"), np.load(validation_filepath+"_Class_Y.npy"), np.load(evaluation_filepath+"_Class_Y.npy")


class _Classifier():
    def __init__(self, model, name):
        self.model = model
        self.name = name
        
    def train(self, X_train, Y_train):
        self.model.fit(X_train, Y_train)
        
    def evaluate(self, X_test, Y_test, out_file=sys.stdout):
        pred = self.model.predict(X_test)
        return classification_report(Y_test, pred), confusion_matrix(Y_test, pred), accuracy_score(Y_test, pred)
        
    def check_sample(self, data):
        index = random.randint(1,data.shape[0]) - 1
        sample = data.iloc[index]
        sample = sample.drop('Sentiment')

        Xnew = [sample]
        ynew = self.model.predict(Xnew)
        print('The sentiment of frame with given parameters is:') 
        print(ynew)

        Xnew = [sample-.35]
        ynew = self.model.predict(Xnew)
        print('The sentiment of frame with given parameters is:') 
        print(ynew)

        Xnew = [sample+.35]
        ynew = self.model.predict(Xnew)
        print('The sentiment of frame with given parameters is:') 
        print(ynew)
        
class DT(_Classifier):
    def __init__(self):
        super().__init__(DecisionTreeClassifier(), "Decision_Tree")

class RFC(_Classifier):
    def __init__(self):
        super().__init__(RandomForestClassifier(n_estimators=200), "Random_Forest_Classifier")
        
class NN(_Classifier):
    def __init__(self):
        super().__init__(MLPClassifier(solver='adam', alpha=1e-5,
                    hidden_layer_sizes=(500, 500, 500), shuffle=True, random_state=1), "Neural_Network")

class NB(_Classifier):#Naive_Bayes
    def __init__(self):
        super().__init__(GaussianNB(), "Naive_Bayes")
    
class SVM(_Classifier):
    def __init__(self):
        super().__init__(svm.SVC(), "Support_Vector_Machine") #LinearSVC
        
class KNN(_Classifier):
    def __init__(self):
        super().__init__(KNeighborsClassifier(n_neighbors = 3), "K_Nearest_Neighbors") #LinearSVC
        
def run_model(classifier, out_file=sys.stdout):
    """ Trains the model with the dance data.
        The History object's History.history attribute is a record of training loss values and metrics values at successive epochs, 
            as well as cooresponding validation values (if applicable).  

    :param model: the model to train
    :type keras.Model
    :param out_file: what to display/write the status information to
    :type output stream
    :return: the class containing the training metric information, the trained model, and the comprehensive evaluation data
    :type tuple
    """
    train_X, train_Y, validate_X, validate_Y, evaluation_X, evaluation_Y, train_Class_Y, validate_Class_Y, evaluation_Class_Y = get_comprehensive_data()
    
    train_X = pd.DataFrame(train_X.reshape((train_X.shape[0], -1))).fillna(0)
    eval_X = pd.DataFrame(evaluation_X.reshape((evaluation_X.shape[0], -1))).fillna(0)
    
    
    train_Class_Y = train_Class_Y.reshape(-1)
    validate_Class_Y = validate_Class_Y.reshape(-1)
    evaluation_Class_Y = evaluation_Class_Y.reshape(-1)

    model = classifier
    model.train(train_X, train_Class_Y)
    classification_report, confusion_matrix, accuracy = model.evaluate(eval_X, evaluation_Class_Y)
    
    #Let's see how our model performed
    print(classification_report)
    write(classification_report, out_file)
    #Confusion matrix for the model
    print("Confusion Matrix:\n"+str(confusion_matrix))
    write("Confusion Matrix:\n"+str(confusion_matrix), out_file)
    print("Accuracy:\t"+str(accuracy))
    write("Accuracy:\t"+str(accuracy), out_file)
    
    