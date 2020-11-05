"""
ECE 236A Project 1, MyClassifier.py template. Note that you should change the
name of this file to MyClassifier_16.py
"""
import pandas as pd
import numpy as np
import cvxpy as cp


class MyClassifier:
    def __init__(self,K,M):
        self.K = K  #Number of classes
        self.M = M  #Number of features
        self.W = []
        self.w = []
        self.classes = []

    #Helper function to find all combinations of classes (one for each hyperplane)
    def get_combos(self):
        combinations = []
        for i in range(self.K-1):
            for j in range(i+1, self.K):
                combinations.append((self.classes[i], self.classes[j]))
        return combinations

    def train(self, p, train_data, train_label):
        print("Training classifier...")
        N_train = train_data.shape[0]
        self.M = train_data.shape[1]
        print("# of inputs: ", N_train)

        #Get number of classes in training data, saving unique class labels
        self.classes = np.sort(np.unique(train_label))
        print("Classes found: ", self.classes)

        #Compute number of classifiers that we need
        self.K = self.classes.size
        L = round(self.K*(self.K-1)/2)

        #Get the pairs of classes compared for each classifier
        combinations = self.get_combos()

        #Initialize parameters
        self.W = np.zeros((self.M, L))
        self.w = np.zeros((1, L))

        #Compute parameters for each classifier we need
        for k in range(L):
            #Get the class labels for this classifier
            class1 = combinations[k][0]
            class2 = combinations[k][1]

            #Get data and labels for the two classes we're comparing
            indices = np.where((train_label == class1) | (train_label == class2))
            valid_labels = train_label[indices]
            valid_data = train_data[indices]

            #Assign labels to 1 or -1
            class1_set = np.where(valid_labels == class1)
            class2_set = np.where(valid_labels == class2)
            valid_labels[class1_set] = 1
            valid_labels[class2_set] = -1

            #Initialize cvxpy parameters
            W = cp.Variable((self.M, 1))
            w = cp.Variable(1)
            lambd = cp.Parameter(nonneg=True)
            m = len(valid_labels)

            valid_labels = valid_labels.reshape(1, len(valid_labels))
            obj = cp.sum(cp.pos(1 - cp.multiply(valid_labels, W.T @ valid_data.T + w)))
            reg = cp.norm(W)
            prob = cp.Problem(cp.Minimize(obj/m + lambd*reg))

            lambd.value = 0.1
            prob.solve()

            self.W[:,k] = np.array(W.value)[:,0]
            self.w[0,k] = w.value
            print("Classifier %d trained." % (k+1))
        
        print("W dimensions: ", self.W.shape)
        print("w dimensions: ", self.w.shape)

        print("Done training.")

        #Project description says to return a MyClassifier object
        return self

    def f(self,input):
        if len(self.W) == 0 or len(self.w) == 0:
            return None
        
        #Compute array of classifications
        #Each element in g represents a 1v1 classifier decision
        g = np.matmul(self.W.T, input) + self.w

        #Store the "votes" for each class: estimates[class] = {count}
        estimates = {}
        pairs = self.get_combos()

        #Check every element in g
        for i in range(g.shape[1]):
            if g[0,i] > 0:
                class_label = pairs[i][0]
            else:
                class_label = pairs[i][1]
            #Increment the count for this class label
            estimates[class_label] = estimates.get(class_label, 0) + 1
        
        #Return the class that got the most votes
        s = None
        max = -1
        for key in estimates:
            if estimates[key] > max:
                max = estimates[key]
                s = key
        return s

    def classify(self,test_data):
        print("Classifying test data...")
        N_test = test_data.shape[0]
        print("# of inputs: ", N_test)
        
        test_results = []
        for image in test_data:
            classification = self.f(image)

            #If f returns None...
            if not classification:
                print("Warning, the classifier is not trained.")
                return None

            test_results.append(classification)

        return test_results

    def TestCorrupted(self,p,test_data):
        #Erase each feature with probability p
        print("Corrupting input data...")
        randn = np.random.rand(*test_data.shape)
        corrupted = np.array(test_data)  #Make a copy to preserve original data
        corrupted[randn < p] = 0
        
        #Call the classify function with the erased data as input
        return self.classify(corrupted)

    #Helper function to check model performance
    def check(self, results, actual):
        num_correct = np.sum(results == actual)
        num_inputs = actual.size
        accuracy = num_correct / num_inputs
        print("%d/%d correct, %.1f%% accuracy\n" % (num_correct, num_inputs, accuracy*100))


if __name__ == "__main__":
    #Load MNIST data
    csv_train = './mnist_train.csv'
    csv_test = './mnist_test.csv'

    #Read CSV files and convert to numpy arrays
    pd_train = pd.read_csv(csv_train)
    pd_test = pd.read_csv(csv_test)

    np_train = pd_train.to_numpy()
    np_test = pd_test.to_numpy()

    print("Label #1:", np_train[0,0])
    print("Image #1 Size:", np_train[0,1:].size)

    train_set_labels = np_train[:,0] 
    train_set = np_train[:,1:]

    test_set_labels = np_test[:,0]
    test_set = np_test[:,1:]

    #Initialize classifier with number of classes and features
    K = 2
    M = train_set.shape[1] 
    classifier = MyClassifier(K,M)

    #Extract training data for only class 1 and class 7
    train_indices = np.where((train_set_labels == 1) | (train_set_labels == 7))
    project_train_labels = train_set_labels[train_indices]
    project_train_data = train_set[train_indices]

    #Extract testing data
    test_indices = np.where((test_set_labels == 1) | (test_set_labels == 7))
    project_test_labels = test_set_labels[test_indices]
    project_test_data = test_set[test_indices]

    #Train classifier
    p = 0.6
    classifier.train(p, project_train_data, project_train_labels)

    #Check results on training data
    print("----Training Data----")
    results = classifier.classify(project_train_data)
    classifier.check(results, project_train_labels)

    #Check results on test data
    print("\n----Test Data----")
    results = classifier.classify(project_test_data)
    classifier.check(results, project_test_labels)

    #Check results on corrupted test data
    print("\n----Corrupted Data----")
    p_vals = [0.4, 0.6, 0.8]
    for i in range(len(p_vals)):
        print("p = %.1f" % p_vals[i])
        results = classifier.TestCorrupted(p_vals[i], project_test_data)
        classifier.check(results, project_test_labels)