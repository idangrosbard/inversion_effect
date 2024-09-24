from enum import Enum

class SampleType(Enum):
    # Classification types
    CLASSIFY_UPRIGHT = 0
    CLASSIFY_INVERTED = 1
    CLASSIFY_BOTH = 2 # not to be used in the datasets

    # Triplet types
    TRIPLET_UPRIGHT = 3
    TRIPLET_INVERTED = 4
    TRIPLET_BOTH = 5 # not to be used in the datasets
    
    # Mixed triplet types
    TRIPLET_MIXED_POS_IMG = 6
    TRIPLET_MIXED_POS_ID = 7