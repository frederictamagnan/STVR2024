from abc import ABC, abstractmethod
class SampleHeuristic(ABC):


    @abstractmethod
    def tests_extraction(self,execution_traces_agilkia_format,cluster_labels):
        pass