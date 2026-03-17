"""
Bazowy moduł oceny dla filmów i użytkowników
"""
from RatingSystem import RatingSystem

class kNN(RatingSystem):
    """
    Klasa reprezentująca działanie Bias Rating System
    """
    def __init__(self):
        super().__init__()

    def __str__(self):
        """
        Ta metoda zwraca numery indeksów wszystkich twórców rozwiązania. Poniżej przykład.
        """
        return 'System created by 155198 and 155921'
