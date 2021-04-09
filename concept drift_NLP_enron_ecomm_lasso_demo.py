import pandas as pd
from scipy.stats import binom_test

#modelop.init
def begin():
    global train, actual_values
    train = pd.read_csv('training_data.csv')
    actual_values = train.flagged
    pass

#modelop.score
def action(datum):
    yield datum

#modelop.metrics
def metrics(data):
    
    predicted_values = data.flagged
    empirical_probability = actual_values.sum()/train.shape[0]
    pvalue = binom_test(predicted_values.sum(), 
                            data.shape[0], 
                            empirical_probability)
    
    yield {"pvalue": pvalue}
