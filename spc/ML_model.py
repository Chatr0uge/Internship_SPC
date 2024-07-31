from sklearn.neighbors import KNeighborsRegressor  
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor, RegressorChain
from sklearn.ensemble import StackingRegressor



class ML_model :
    
    def __init__(self, model_names : str, stacking : bool = True, **kwargs) :
        
        self.model_names = model_names
        self.kwargs = kwargs
        self.stacking = stacking
        self.dict_model = {"KNN" : KNeighborsRegressor(), "GP" : GaussianProcessRegressor, "RF" : RandomForestRegressor(max_depth=20, n_estimators=200, criterion='absolute_error'), "Multi" : MultiOutputRegressor(GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, loss = 'absolute_error')), "Chain" : RegressorChain(GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, loss = 'absolute_error'))}

        self.model = self.get_model()
        
    def get_model(self) :
        
        for model_name in self.model_names :
            if model_name not in ["KNN", "GP", "RF", "Multi", "Chain"] :
                raise ValueError("Model name not recognized")
        if self.stacking :
            m = StackingRegressor(estimators = [(model_name, self.dict_model[model_name]) for model_name in self.model_names],  stack_method_='predict',**self.kwargs,)
            
        else :
            m = [self.dict_model[model_name] for model_name in self.model_names]
            
        return m
    def fit(self, X, y) :
        if self.stacking :
            self.model.fit(X, y)
        else :
            for model in self.model :
                model.fit(X, y)
                
    def predict(self, X) :
        if self.stacking :
            return self.model.predict(X)
        else :
            return np.array([model.predict(X) for model in self.model])
    
    def score(self, X, y) :
        if self.stacking :
            return self.model.score(X, y)
        else :
            return np.array([model.score(X, y) for model in self.model])
    
    
    