from enum import Enum


class ClientTypes(Enum):
    FED_AVG = 1
    FED_PROX = 2
    FED_BN = 3
    FED_MRI = 4


class LossFunctions(Enum):
    MSE = 1
    MSE_DSSIM = 2
    PROX = 3
    RMSE_DDSIM = 4
    MSE_ZOOMED_DSSIM = 5


class AggregationMethods(Enum):
    FED_AVG = 1
    FED_PROX = 2
    FED_ADAM = 3
    FED_ADAGRAD = 4
    FED_YOGI = 5
    FED_COSTW = 6
    FED_PID = 7
    FED_AVGM = 8
    FED_MEAN = 9
    FED_TRIMMED = 10


class NormalizationType(Enum):
    BN = 1
    GN = 2
    NONE = 3


class ImageModality(Enum):
    TUMOR = -1
    MASK = 0
    T1 = 1
    T2 = 2
    FLAIR = 3
