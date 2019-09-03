from abc import ABC, abstractmethod

class BaseModel(ABC):
  """
  Defines an interface for all tested models to implement.

  :param rouge_scores:    Mean rouge scores over 
  :param name:            Model's name.
  """
  def __init__(self, name):
    self.mean_r_scores = None
    self.name = name
    self.fail = []

  @abstractmethod
  def transform(self, train_x):
    ...

  @abstractmethod
  def score(self, test_x, test_y):
    ...

  @abstractmethod
  def report(self, pred_y, test_y):
    ...
    
class ITrainableModel(BaseModel):
  @abstractmethod
  def fit(self, train_x, train_y):
    ...

  @abstractmethod
  def fit_transform(self, train_x, train_y):
    ...