# Metoda nr 1 - BIAS MODEL
## Opis podejścia - Bias-Based Collaborative Filtering
Głównym założeniem tej metody jest system rekomendacji opierający się na odchyleniach oraz ukrytych cechach użytkowników i filmów. Zamiast zwykłego wyciągania średniej, przewidujemy ocenę ucząc model algorytmem SGD przez kilka epok, minimalizując błąd predykcji z regularyzacją L2. Końcowa predykcja składa się z czterech elementów:
- globalnej średniej ocen wszystkich filmów,
- biasu użytkownika – czy recenzent jest surowy czy hojny w ocenach,
- biasu filmu – czy film jest obiektywnie oceniany wyżej lub niżej od średniej,
- iloczynu skalarnego wektorów ukrytych cech użytkownika i filmu – czyli wyuczonego dopasowania preferencji użytkownika do charakterystyki filmu.

## Funkcje z pliku system155198.py
`calculate_global_mean` - odniesienie do całego systemu, czyli wyliczenie całości średniej globalnej.
`calculate_user_bias` - sprawdzenie, o ile średnia ocen konretnego usera różni się od średniej globalnej.
`calculate_movie_bias` - to samo, tylko dotyczy filmów
`clamp_raiting` - tutaj znajduje sie zabezpieczenie, tzn. suma biasów nie może wyskoczyć poza naszą skalę.
`rate` - zlicza względem naszej funkcji: średnia + bias usera + bias użytkownika. Główny punkt algorytmu.
`_get_train_data` – zbiera wszystkie oceny użytkowników do listy krotek (user_id, movie_id, rating) i inicjalizuje losowe wektory cech oraz biasy dla każdego użytkownika i filmu.
`_train` – trenuje model przez kilka epok algorytmem SGD, iteracyjnie poprawiając biasy i wektory cech użytkowników i filmów na podstawie błędu predykcji.

## Znalezione źródła
[Artykuł PMC - 10 mar 2026]
https://pmc.ncbi.nlm.nih.gov/articles/PMC12208497/
- Zbliżona metoda do naszego podejścia, pokazuje że metody korygujące błąd (u nas lambda), dają relanie lepsze rekomendacje niż czyste systemy bez korekty

[Artykuł ARXIV - 11 mar 2026]
https://arxiv.org/pdf/2203.00376
- Podobny realizowany problem implementacji rekomendacji oraz wybrane sposoby radzenia sobie z nimi - skłonności użytkownika do popularności/jego preferencji

[Artykuł CEUR-WS - 11 mar 2026]
https://ceur-ws.org/Vol-2440/paper6.pdf
- Analiza problemu zamiast wzorów, podejście bardziej pod kątem koncepcyjnym

[Artykuł ACM - 12 mar 2026]
https://dl.acm.org/doi/abs/10.1145/3437963.3441820
- Artykuł ten stanowi bardziej rozwinięta wersje naszego podejścia