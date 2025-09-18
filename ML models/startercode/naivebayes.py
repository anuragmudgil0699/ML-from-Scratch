"""
 Refer to Linear Models notes or Chapter 4 of J&M for more details on how to implement a Naive Bayes Model
"""

from Model import *
class NaiveBayes(Model):
    
    def train(self, input_file):
        """
        This method is used to train your model and will generate a trained model file for some given input_file
        :param input_file: path to training file with a text and a label per each line
        :return: model: trained model 
        """
        ## TODO write your code here
        x = 3
        breakpoint()
        model = None
        ## Save the model
        self.save_model(model)
        return model


    def classify(self, input_file, model):
        """
        This method will be called by us for the validation stage and or you can call it for evaluating your code 
        on your own splits that you make from the training sets given to you
        :param input_file: path to input file with a text per line **without labels** (note that this is different from the training data format!)
        :param model: the pretrained model
        :return: predictions list
        """
        ## TODO write your code here
        preds = None
        return preds


